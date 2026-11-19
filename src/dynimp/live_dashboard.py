"""Live dashboard for dynimp D1 cells (Option A): tune gap-filling methods on old data, no retraining.

Run from the repo root:

    PYTHONPATH=src .venv-dynimp/Scripts/python.exe -m dynimp.live_dashboard [--port 8765]

Then open http://127.0.0.1:8765/ in a browser.

Scope: the 10 D1 cells only (Lorenz / Rossler / Mackey-Glass x point/block, BIDMC PPG g62/g187,
ECG g25/g75). Units are the D1 test units: chaotic seeds 0-5, BIDMC test patients (20).

What is live (recomputed per request, CPU, on the viewed segment only):
  * recurrence fill, D3 engine (dynimp.fills_v2): rule, C (single or averaged over a C set), k, ECG weight
  * recurrence fill, D1 engine (dynimp.fills.recurrence_fill): C, k, min_overlap, channel subset
  * linear / spline interpolation, low-rank Hankel completion, mean fill

What is frozen (inference only, never refit):
  * SAITS_cross / CSDI_cross for the BIDMC cells: model states are loaded from the D2 stage_cross
    caches and only predict on the current test windows. Fine-tuned / own-patient variants kept no
    states, so they appear only as stored scores. Chaotic SAITS/CSDI also kept no states or
    predictions, so no learned curve is shown for chaotic cells (stored scores only).

The RP panel compares, for the viewed window, the unthresholded recurrence plot of each filled
series against the truth RP (relative Frobenius error, as in d1.rp_error), using each system's
scored embedding. RP matrices are sent at full resolution up to 512x512 states; only longer windows
are block-averaged down to that size for display.
"""
from __future__ import annotations

import argparse
import json
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from dynimp import d0, d1, d3
from dynimp.fills import lowrank_fill_gaps, lowrank_fill_tiled, recurrence_fill, spline_fill
from dynimp.fills_v2 import RULES, recurrence_fills_v2
from condpred import completion
from recplay.embedding import embed

REPO = Path(__file__).resolve().parents[2]
RP_MAX = 512          # display resolution of the RP heatmaps (no averaging below this many states)
EXPAND = 300          # context samples added on both sides of the viewed window before filling

D1SPEC = json.loads((REPO / "notebooks" / "tables" / "dynimp" / "D1" / "spec.json").read_text())
D3SPEC = json.loads((REPO / "notebooks" / "tables" / "dynimp" / "D3" / "spec.json").read_text())
D2SPEC = json.loads((REPO / "notebooks" / "tables" / "dynimp" / "D2" / "spec.json").read_text())
CELLS = D1SPEC["cells"]
CHAOTIC_EMB = {k: tuple(v) for k, v in d0.EMBEDDING.items()}

_series_cache: dict = {}
_learned_cache: dict = {}
_cross_cache: dict = {}


# ----------------------------------------------------------------------------- data

def parse_cell(cell: str):
    if "|n0.1|" in cell:
        system, pattern = cell.split("|")[0], cell.split("|")[-1]
        return ("chaotic", system, pattern, None)
    task, g = cell.split("|")
    return ("physio", task, None, int(g[1:]))


def embedding_for(cell: str) -> tuple[int, int]:
    kind, name, _, _ = parse_cell(cell)
    if kind == "chaotic":
        return CHAOTIC_EMB[name]
    return tuple(D1SPEC["embedding"][name])


def units_for(cell: str) -> list[int]:
    kind, *_ = parse_cell(cell)
    return list(range(6)) if kind == "chaotic" else [int(s) for s in D1SPEC["subjects"]["test"]]


def load_series(cell: str, unit: int) -> dict:
    """Truth / observed / mask for the scored (test) segment of one unit, plus the donor library."""
    key = (cell, int(unit))
    if key in _series_cache:
        return _series_cache[key]
    kind, name, _, g = parse_cell(cell)
    if kind == "physio":
        p = d3.physio_prep(name, g, int(unit))
        out = {"kind": kind, "task": name, "g": g, "truth_all": p["truth"],  # target channel only
               "scored": p["scored"], "miss": p["miss"], "hist": p["hist"],
               "channels": list(d1.TASKS[name]["columns"]), "target": 0}
    else:
        p = d3.chaotic_prep(cell, int(unit))
        out = {"kind": kind, "system": name, "truth_all": p["truth"], "scored": p["scored"],
               "miss": p["miss"], "hist": p["hist"], "channels": ["x", "y", "z"][: p["F"]], "target": 0}
    _series_cache.clear() if len(_series_cache) > 3 else None
    _series_cache[key] = out
    return out


def gap_runs(miss: np.ndarray) -> list[list[int]]:
    m = np.asarray(miss, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return [[int(a), int(b)] for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]




# ----------------------------------------------------------------------------- RP

def distance_matrix(series: np.ndarray, m: int, tau: int) -> np.ndarray:
    """RP distance matrix of one window, z-scored as in d1.rp_error."""
    mu, sd = float(np.mean(series)), float(np.std(series)) or 1.0
    E = embed((np.asarray(series, float) - mu) / sd, m, tau)
    return np.sqrt(((E[:, None, :] - E[None, :, :]) ** 2).sum(-1))


def downsample(M: np.ndarray) -> np.ndarray:
    """Block-average a matrix to at most RP_MAX x RP_MAX for display."""
    if len(M) <= RP_MAX:
        return M
    k = int(np.ceil(len(M) / RP_MAX))
    n = (len(M) // k) * k
    return M[:n, :n].reshape(n // k, k, n // k, k).mean(axis=(1, 3))


# ----------------------------------------------------------------------------- frozen learned models

def _cross(cell: str) -> dict:
    if cell in _cross_cache:
        return _cross_cache[cell]
    _, task, _, g = parse_cell(cell)
    files = sorted((REPO / "data" / "processed" / "dynimp" / "D2" / "full").glob(f"stage_cross_{task}_g{g}_*.pkl"))
    if not files:
        raise RuntimeError(f"no D2 cross cache for {cell}")
    out = pickle.loads(files[0].read_bytes())
    _cross_cache[cell] = out
    return out


def learned_cross(cell: str, unit: int, scored: np.ndarray, which: str) -> np.ndarray:
    """Target-channel predictions of a frozen D2 cross model on non-overlapping 375-sample windows."""
    key = (cell, int(unit), which, hash(scored.tobytes()))
    if key in _learned_cache:
        return _learned_cache[key]
    from dynimp import d2

    cross = _cross(cell)
    L = D2SPEC["tasks"][parse_cell(cell)[1]]["L"]
    wins = d2.win(np.asarray(scored, float), L, L)
    if which == "saits_cross":
        model = d2._saits(cross["chosen_saits"], L, wins.shape[2], 1, 1)
        model.model.load_state_dict(cross["states"]["saits"])
        imp = np.asarray(model.predict({"X": wins})["imputation"], float)
    else:
        model = d2._csdi(L, wins.shape[2], 1, 1)
        model.model.load_state_dict(cross["states"]["csdi"])
        samples = np.asarray(model.predict({"X": wins}, n_sampling_times=D2SPEC["csdi_n_samples"])["imputation"], float)
        imp = np.median(samples, axis=1)
    pred = np.full(len(scored), np.nan)
    pred[: imp.shape[0] * L] = imp[:, :, 0].reshape(-1)
    _learned_cache.clear() if len(_learned_cache) > 4 else None
    _learned_cache[key] = pred
    return pred


# ----------------------------------------------------------------------------- filling

def compute_fills(cell: str, unit: int, channel: int, start: int, length: int, params: dict) -> dict:
    s = load_series(cell, unit)
    scored, hist = s["scored"], s["hist"]
    n = len(scored)
    start, length = int(start), int(length)
    lo, hi = max(0, start - EXPAND), min(n, start + length + EXPAND)
    seg = np.array(scored[lo:hi], float)          # observed, multivariate, NaN at gaps
    truth_seg_all = np.asarray(s["truth_all"][lo:hi], float)
    miss_seg_all = np.asarray(s["miss"][lo:hi])

    if s["kind"] == "chaotic":
        c = int(channel)
        truth_seg, miss_seg = truth_seg_all[:, c], miss_seg_all[:, c]
    else:
        c = 0
        truth_seg, miss_seg = truth_seg_all, miss_seg_all


    v_lo, v_hi = start - lo, start - lo + length   # viewed window inside the segment
    methods = set(params.get("methods", []))
    fills: dict[str, np.ndarray] = {}

    # --- recurrence, D3 engine (fills_v2) ---
    if "recurrence_v2" in methods:
        Cs = [int(x) for x in params.get("Cs", [int(params.get("C", 20))])]
        w = float(params.get("w", 1.0))
        own_only = params.get("helper_mode") == "own" and seg.shape[1] > 1
        tseg = seg[:, [c]] if own_only else seg
        tlib = hist[:, [c]] if own_only else hist
        cc = 0 if own_only else c
        wv = {w: np.array([1.0, w])} if (seg.shape[1] == 2 and not own_only) else {1.0: np.ones(tseg.shape[1])}
        wl = w if (seg.shape[1] == 2 and not own_only) else 1.0
        acc = []
        for C in Cs:
            r = recurrence_fills_v2(tseg, tlib, cc, C=C, rules=[params.get("rule", "ctx20")],
                                    weight_vectors=wv, k=int(params.get("k", 8)))
            acc.append(r["fills"][(params.get("rule", "ctx20"), wl)])
        fills["RP fill_v2"] = np.nanmean(np.stack(acc), axis=0) if len(acc) > 1 else acc[0]

    # --- recurrence, D1 engine ---
    if "recurrence_d1" in methods:
        mode = params.get("helper_mode", "all")
        if seg.shape[1] > 1 and mode == "own":
            cols, wts = [c], None
        elif seg.shape[1] > 1 and mode == "helper":
            cols, wts = list(range(seg.shape[1])), [0.0] * seg.shape[1]
            wts[c] = 0.0
            wts[1 - c if seg.shape[1] == 2 else 1] = 1.0
        else:
            cols, wts = list(range(seg.shape[1])), None
        fills["RP fill"] = recurrence_fill(seg[:, cols], hist[:, cols], cols.index(c),
                                                C=int(params.get("C", 20)), k=int(params.get("k", 8)),
                                                min_overlap=float(params.get("min_overlap", 0.3)), weights=wts)

    # --- simple methods on the target channel ---
    x = seg[:, c]
    if "linear" in methods:
        fills["linear"] = completion.linear(x)
    if "spline" in methods:
        fills["spline"] = spline_fill(x)
    if "lowrank" in methods:
        if s["kind"] == "chaotic":
            fills["low-rank"] = lowrank_fill_tiled(x, int(params.get("lr_window", 96)), int(params.get("lr_rank", 4)))
        else:
            fills["low-rank"] = lowrank_fill_gaps(x, int(params.get("lr_window", 250)), int(params.get("lr_rank", 8)))
    if "mean" in methods:
        fills["mean"] = np.zeros(len(x))

    # --- frozen learned models (physio only), aligned to the segment ---
    learned_notes = []
    for which, label in [("saits_cross", "SAITS"), ("csdi_cross", "CSDI")]:
        if which in methods:
            if s["kind"] != "physio":
                learned_notes.append(f"{label}: no stored model states or predictions for chaotic cells; not shown without retraining.")
            else:
                full = np.array(scored, float)
                pred = learned_cross(cell, unit, full, which)
                fills[label] = pred[lo:hi]

    # --- metrics + RP over the viewed window ---
    m_emb, tau = embedding_for(cell)
    truth_view = truth_seg[v_lo:v_hi]
    miss_view = miss_seg[v_lo:v_hi]
    filled_truth = truth_view.copy()
    D_true = distance_matrix(truth_view, m_emb, tau)
    off = ~np.eye(len(D_true), dtype=bool)
    eps = float(np.quantile(D_true[off], 0.05)) if off.any() else 0.0
    B_true = D_true <= eps
    out_methods = []
    for label, f in fills.items():
        fv = np.asarray(f, float)[v_lo:v_hi]
        mae = float(np.abs(np.nan_to_num(fv[miss_view], nan=0.0) - truth_view[miss_view]).mean()) if miss_view.any() else None
        filled = np.where(miss_view, np.nan_to_num(fv, nan=0.0), truth_view)
        D = distance_matrix(filled, m_emb, tau)
        rp_err = float(np.linalg.norm(D - D_true) / max(np.linalg.norm(D_true), 1e-300))
        B_fill = D <= eps
        joint = B_true & B_fill
        out_methods.append({"label": label, "fill": _round(fv), "mae_view": mae, "rp_error": rp_err,
                            "unfilled": int((~np.isfinite(fv[miss_view])).sum()),
                            "rp_filled": _round(downsample(B_fill.astype(float))),
                            "rp_joint": _round(downsample(joint.astype(float))),
                            "joint_share": float(joint.sum() / max(int(B_true.sum()), 1))})

    return {"start": start, "length": length, "x": list(range(start, start + length)),
            "truth": _round(truth_view), "observed": _round(np.where(miss_view, np.nan, truth_view)),
            "miss": miss_view.astype(int).tolist(),
            "helper": _round(seg[v_lo:v_hi, 1]) if (s["kind"] == "physio" and seg.shape[1] > 1) else None,
            "methods": out_methods, "rp_true": _round(downsample(B_true.astype(float))),
            "rp_eps": eps, "embedding": [m_emb, tau],
            "notes": learned_notes}


def _round(a) -> list:
    a = np.asarray(a, float)
    return [None if not np.isfinite(v) else round(float(v), 4) for v in a.ravel()] if a.ndim == 1 else \
        [[None if not np.isfinite(v) else round(float(v), 4) for v in row] for row in a]


# ----------------------------------------------------------------------------- HTTP

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Gap-fill dashboard</title>
<script src="/vendor/plotly-2.35.2.min.js"></script>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#fafafa;color:#222}
 header{padding:10px 16px;background:#fff;border-bottom:1px solid #ddd;display:flex;flex-wrap:wrap;gap:12px;align-items:center}
 header h1{font-size:15px;margin:0 10px 0 0}
 label{font-size:12.5px} select,input{font-size:12.5px;margin-left:3px}
 .bar{padding:7px 16px;background:#fff;border-bottom:1px solid #eee;font-size:12.5px;display:flex;flex-wrap:wrap;gap:14px;align-items:center}
 .row{display:grid;grid-template-columns:1fr 1fr;gap:8px;padding:8px}
 .panel{background:#fff;border:1px solid #ddd;border-radius:4px;padding:6px}
 .note{font-size:12px;color:#555;padding:2px 4px 6px}
 .tabs{padding:8px 8px 0;display:flex;gap:6px}
 .tabbtn{font-size:13px;padding:5px 14px;border:1px solid #bbb;border-bottom:none;background:#eee;border-radius:4px 4px 0 0;cursor:pointer}
 .tabbtn.active{background:#fff;font-weight:600}
 #status{color:#555}
</style></head><body>
<header>
 <h1>Gap-fill dashboard</h1>
 <label>Cell <select id="cell"></select></label>
 <label>Unit <select id="unit"></select></label>
 <label>Channel <select id="channel"></select></label>
 <label>Window start <input id="start" type="number" value="0" step="50" style="width:80px"></label>
 <label>Length <select id="length"><option>250</option><option selected>500</option><option>1000</option><option>2000</option></select></label>
 <button id="prevGap">← prev gap</button><button id="nextGap">next gap →</button>
 <button id="settingsBtn">Hide settings</button>
 <span id="status"></span>
</header>
<div class="bar">
 <b>RP fill_v2:</b>
 <label>rule <select id="rule"><option>old</option><option>old_fb</option><option>ctx10</option><option selected>ctx20</option><option>ctx40</option><option>ctx60</option><option>pen05</option><option>pen10</option></select></label>
 <label>C <input id="C" type="number" value="20" min="1" max="250" style="width:55px"></label>
 <label>average over Cs <input id="Cs" type="text" value="" placeholder="e.g. 5,10,20" style="width:90px"></label>
 <label>k <input id="k" type="number" value="8" min="1" max="32" style="width:45px"></label>
 <label>ECG weight w <input id="w" type="number" value="1" step="0.25" min="0" max="2" style="width:50px"></label>
 <label>helper <select id="helper_mode"><option value="all">all channels</option><option value="own">target only</option><option value="helper">helper only</option></select></label>
 <label>min_overlap - RP fill <input id="min_overlap" type="number" value="0.3" step="0.05" min="0" max="1" style="width:50px"></label>
 <button id="d3preset">D3 final preset</button>
</div>
<div class="bar">
 <b>Methods:</b>
 <label><input type="checkbox" class="meth" value="recurrence_v2" checked> RP fill_v2</label>
 <label><input type="checkbox" class="meth" value="recurrence_d1" checked> RP fill</label>
 <label><input type="checkbox" class="meth" value="linear" checked> linear</label>
 <label><input type="checkbox" class="meth" value="spline" checked> spline</label>
 <label><input type="checkbox" class="meth" value="lowrank"> low-rank</label>
 <label><input type="checkbox" class="meth" value="mean"> mean</label>
 <label><input type="checkbox" class="meth" value="saits_cross"> SAITS</label>
 <label><input type="checkbox" class="meth" value="csdi_cross"> CSDI</label>
 <label>low-rank rank <input id="lr_rank" type="number" value="8" min="1" max="24" style="width:45px"></label>
 <label>low-rank window/ctx <input id="lr_window" type="number" value="250" min="10" max="500" style="width:55px"></label>
</div>
<div class="bar">
 <button id="apply">Apply / recompute</button>
</div>
<div class="tabs">
 <button class="tabbtn active" id="tabbtn-series">Series</button>
 <button class="tabbtn" id="tabbtn-rp">Recurrence plots</button>
</div>
<div id="tab-series">
<div class="row" style="grid-template-columns:1fr">
 <div class="panel"><div id="series" style="width:100%;aspect-ratio: 3/ 1"></div><div id="helperPlot" style="height:120px"></div></div>
 <div class="panel"><div id="errplot" style="width:100%;aspect-ratio: 3/ 1"></div><div class="note">Absolute error inside gaps only. Lines outside gaps are hidden (every method equals the truth there).</div></div>
</div>
</div>
<div id="tab-rp" style="display:none">
<div class="row" style="grid-template-columns:1fr">
 <div class="panel"><div id="rpplots"></div><div class="note" id="rpnote"></div></div>
</div>
</div>
<script>
const $=id=>document.getElementById(id);
let META=null, SERIES=null, FILL=null;
async function jget(u){const r=await fetch(u); if(!r.ok) throw new Error(await r.text()); return r.json();}
async function jpost(u,b){const r=await fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(b)}); if(!r.ok) throw new Error(await r.text()); return r.json();}
function params(){
  const meth=[...document.querySelectorAll(".meth:checked")].map(e=>e.value);
  const Cs=$("Cs").value.trim()? $("Cs").value.split(",").map(x=>parseInt(x)).filter(x=>x>0):null;
  const p={methods:meth, rule:$("rule").value, C:parseInt($("C").value), k:parseInt($("k").value),
    w:parseFloat($("w").value), helper_mode:$("helper_mode").value, min_overlap:parseFloat($("min_overlap").value),
    lr_rank:parseInt($("lr_rank").value), lr_window:parseInt($("lr_window").value)};
  if(Cs) p.Cs=Cs;
  return p;
}
async function loadSeries(){
  $("status").textContent="loading series…";
  SERIES=await jget(`/api/series?cell=${encodeURIComponent($("cell").value)}&unit=${$("unit").value}&channel=${$("channel").value}`);
  $("start").max=SERIES.n-1;
  $("status").textContent=`test length ${SERIES.n}, ${SERIES.gaps.length} gap runs, embedding (m,tau)=(${SERIES.embedding})`;
  await recompute();
}
async function recompute(){
  $("status").textContent="computing…";
  try{
    FILL=await jpost("/api/fill",{cell:$("cell").value, unit:parseInt($("unit").value), channel:parseInt($("channel").value),
      start:parseInt($("start").value), length:parseInt($("length").value), params:params()});
    render();
    $("status").textContent="done";
  }catch(e){ $("status").textContent="error: "+e.message; }
}
function gapOnly(fill,miss){ return fill.map((v,i)=> miss[i]? v : (i>0&&miss[i-1])||(i<fill.length-1&&miss[i+1])? v : null); }
function render(){
  const x=FILL.x, traces=[
    {x, y:FILL.truth, name:"truth", line:{color:"black",width:2}},
    {x, y:FILL.observed, name:"observed", line:{color:"#999",width:1}}];
  const shapes=[];
  let inGap=false, gs=0;
  FILL.miss.forEach((m,i)=>{ if(m&&!inGap){gs=x[i];inGap=true;} if(!m&&inGap){shapes.push({type:"rect",xref:"x",yref:"paper",x0:gs,x1:x[i],y0:0,y1:1,fillcolor:"rgba(220,50,50,0.10)",line:{width:0}});inGap=false;} });
  if(inGap) shapes.push({type:"rect",xref:"x",yref:"paper",x0:gs,x1:x[x.length-1],y0:0,y1:1,fillcolor:"rgba(220,50,50,0.10)",line:{width:0}});
  const etraces=[];
  FILL.methods.forEach(mt=>{
    traces.push({x, y:gapOnly(mt.fill,FILL.miss), name:mt.label, line:{width:1.6}});
    etraces.push({x, y:mt.fill.map((v,i)=>FILL.miss[i]&&v!=null?Math.abs(v-FILL.truth[i]):null), name:mt.label});
  });
  Plotly.newPlot("series",traces,{title:"truth / reconstructions",shapes,margin:{t:35,l:45,r:10,b:35},legend:{orientation:"h"}},{responsive:true});
  Plotly.newPlot("errplot",etraces,{title:"error",margin:{t:35,l:45,r:10,b:35},legend:{orientation:"h"}},{responsive:true});
  if(FILL.helper){ Plotly.newPlot("helperPlot",[{x,y:FILL.helper,name:"helper ECG",line:{color:"#666"}}],{title:"helper channel",margin:{t:25,l:45,r:10,b:25}},{responsive:true}); }
  else $("helperPlot").innerHTML="";
  renderRP();
}
function renderRP(){
  if(!FILL || $("tab-rp").style.display==="none") return;
  const rp=$("rpplots"); rp.innerHTML="";
  const mk=(parent,data,title,id)=>{
    const d=document.createElement("div"); d.id=id; d.style.width="100%"; d.style.aspectRatio="1 / 1"; parent.appendChild(d);
    // winter: matplotlib colour stops inlined (Plotly has no scale with this name)
    Plotly.newPlot(id,[{z:data,type:"heatmap",zmin:0,zmax:1,colorscale:[[0,"#0000ff"],[0.1,"#0019f2"],[0.2,"#0033e6"],[0.3,"#004cd9"],[0.4,"#0066cc"],[0.5,"#0080bf"],[0.6,"#0099b2"],[0.7,"#00b3a6"],[0.8,"#00cc99"],[0.9,"#00e68c"],[1,"#00ff80"]],showscale:false}],{title,margin:{t:30,l:35,r:10,b:30},yaxis:{scaleanchor:"x",scaleratio:1}},{responsive:true});
  };
  FILL.methods.forEach((mt,i)=>{
    const h=document.createElement("h3"); h.style.margin="10px 0 4px"; h.textContent=`${mt.label} - RP error ${mt.rp_error.toFixed(4)}`; rp.appendChild(h);
    const row=document.createElement("div");
    row.style.display="grid"; row.style.gridTemplateColumns="repeat(3,1fr)"; row.style.gap="8px"; row.style.marginBottom="14px";
    rp.appendChild(row);
    mk(row,FILL.rp_true,"original system",`rp_${i}_orig`);
    mk(row,mt.rp_filled,mt.label,`rp_${i}_fill`);
    mk(row,mt.rp_joint,`joint plot ~ ${(100*mt.joint_share).toFixed(1)}% of original`,`rp_${i}_joint`);
  });
  $("rpnote").textContent=`Binary recurrence plots over the viewed window: state i,j is recurrent when the embedded distance is at most epsilon = ${FILL.rp_eps.toFixed(4)}, chosen so the original system has recurrence rate 5% (embedding (m, tau) = (${FILL.embedding}), window z-scored by the original). The same epsilon is used for every method; joint = recurrent in both the original and the filled system. Matrices are shown at full resolution (one cell per state pair) up to 512 states; longer windows are block-averaged down to 512x512. RP error is computed on the full unthresholded distance matrices. `+(FILL.notes||[]).join(" ");
}
function showTab(name){
  $("tab-series").style.display = name==="series"? "" : "none";
  $("tab-rp").style.display = name==="rp"? "" : "none";
  $("tabbtn-series").classList.toggle("active", name==="series");
  $("tabbtn-rp").classList.toggle("active", name==="rp");
  if(name==="rp" && FILL) renderRP();
  if(name==="series") window.dispatchEvent(new Event("resize"));
}
function jumpGap(dir){
  if(!SERIES || !SERIES.gaps.length) return;
  const len=parseInt($("length").value);
  const off=Math.floor(len/4);
  const anchor=parseInt($("start").value)+off;  // gap the current window is positioned on
  const starts=SERIES.gaps.map(g=>g[0]);
  let nxt = dir>0? starts.find(s=>s>anchor) : [...starts].reverse().find(s=>s<anchor);
  if(nxt===undefined) nxt = dir>0? starts[0] : starts[starts.length-1];
  $("start").value=Math.max(0, nxt-off);
  recompute();
}
(async()=>{
  META=await jget("/api/meta");
  META.cells.forEach(c=>$("cell").add(new Option(c,c)));
  $("cell").value="ppg|g62";
  const syncUnits=()=>{
    $("unit").innerHTML=""; META.units[$("cell").value].forEach(u=>$("unit").add(new Option(u,u)));
    $("channel").innerHTML=""; META.channels[$("cell").value].forEach((c,i)=>$("channel").add(new Option(c,i)));
  };
  syncUnits();
  $("cell").onchange=()=>{syncUnits(); $("start").value=0; loadSeries();};
  $("unit").onchange=loadSeries; $("channel").onchange=loadSeries;
  $("apply").onclick=recompute;
  ["length"].forEach(id=>$(id).onchange=recompute);
  $("nextGap").onclick=()=>jumpGap(1); $("prevGap").onclick=()=>jumpGap(-1);
  $("tabbtn-series").onclick=()=>showTab("series");
  $("tabbtn-rp").onclick=()=>showTab("rp");
  $("settingsBtn").onclick=()=>{
    const bars=[...document.querySelectorAll(".bar")];
    const hide=bars[0].style.display!=="none";
    bars.forEach(b=>b.style.display=hide? "none" : "");
    $("settingsBtn").textContent=hide? "Show settings" : "Hide settings";
    window.dispatchEvent(new Event("resize"));
  };
  $("d3preset").onclick=()=>{
    const cell=$("cell").value, ch=META.d3_choices[cell];
    if(!ch) return;
    $("rule").value="ctx20"; $("Cs").value=ch.pool_set.join(","); $("C").value=ch.pool_C; $("w").value=ch.w;
    recompute();
  };
  await loadSeries();
})();
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep console quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            elif u.path == "/vendor/plotly-2.35.2.min.js":
                self._send(200, (REPO / "dashboard" / "vendor" / "plotly-2.35.2.min.js").read_bytes(), "text/javascript")
            elif u.path == "/api/meta":
                meta = {"cells": CELLS,
                        "units": {c: units_for(c) for c in CELLS},
                        "channels": {}, "d3_choices": D3SPEC.get("per_cell_choices", {})}
                for c in CELLS:
                    kind, name, _, _ = parse_cell(c)
                    meta["channels"][c] = ["x", "y", "z"] if kind == "chaotic" else list(d1.TASKS[name]["columns"])
                self._send(200, json.dumps(meta))
            elif u.path == "/api/series":
                q = parse_qs(u.query)
                cell, unit, channel = q["cell"][0], int(q["unit"][0]), int(q.get("channel", ["0"])[0])
                s = load_series(cell, unit)
                if s["kind"] == "chaotic":
                    miss = np.asarray(s["miss"])[:, channel]
                else:
                    miss = np.asarray(s["miss"])
                self._send(200, json.dumps({"n": int(len(miss)), "gaps": gap_runs(miss),
                                            "embedding": list(embedding_for(cell))}))
            else:
                self._send(404, json.dumps({"error": "not found"}))
        except Exception as e:  # surface to the UI status line
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))

    def do_POST(self):
        u = urlparse(self.path)
        try:
            if u.path != "/api/fill":
                self._send(404, json.dumps({"error": "not found"}))
                return
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            out = compute_fills(body["cell"], int(body["unit"]), int(body.get("channel", 0)),
                                int(body["start"]), int(body["length"]), body.get("params", {}))
            self._send(200, json.dumps(out))
        except Exception as e:
            self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dynimp live dashboard: http://{args.host}:{args.port}/  (Ctrl-C to stop)", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
