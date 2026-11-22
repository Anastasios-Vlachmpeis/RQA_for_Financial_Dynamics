"""Build dashboard/torp_triplets.html: each financial triplet in 3D (return cloud and thresholded perimeter TORP).

Run from this folder after 01_triplet_perimeter.ipynb:  python triplet_dashboard.py
Reads the notebook's price cache and per-window table; the page recomputes states and tensor cells in the
browser with the notebook's ε, so it needs no server. Plotly.js is downloaded once into dashboard/vendor/.
"""
import json
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

PRICES = Path("../../data/raw/yahoo/triplet_prices_2015_2024.csv")
WINDOWS = Path("tables/torp_triplets/triplet_torp_windows.csv")
DASHBOARD = Path("dashboard")
PLOTLY = DASHBOARD / "vendor" / "plotly-2.35.2.min.js"
TEMPLATE = Path(__file__).with_name("triplet_dashboard_template.html")
M, TAU, W, RR = 3, 1, 250, 0.05     # must match 01_triplet_perimeter


def main():
    prices = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    returns = np.log(prices).diff().iloc[1:].dropna()
    cases = pd.read_csv(WINDOWS)
    dates = [d.strftime("%Y-%m-%d") for d in returns.index]

    windows = []
    for w_id, g in cases.groupby("window"):
        start = dates.index(g["start"].iloc[0])
        windows.append({"start": start, "label": f"{g['start'].iloc[0]} → {g['end'].iloc[0]}"})
    assert all(w["start"] == i * W for i, w in enumerate(windows))

    variants = []
    stats = ["eps", "D_all", "sync_ratio", "predicted_ratio", "sync_enrichment", "RR_sync", "r_ab", "r_ac", "r_bc"]
    for (category, variant), g in cases.groupby(["category", "variant"], sort=False):
        tickers = [g["a"].iloc[0], g["b"].iloc[0], g["c"].iloc[0]]
        sign = [-1 if (variant == "−VIX" and t == "^VIX") else 1 for t in tickers]
        variants.append({"category": category, "variant": variant, "tickers": tickers, "sign": sign,
                         "windows": g.sort_values("window")[stats].round(6).to_dict("records")})

    data = {"M": M, "TAU": TAU, "W": W, "RR": RR, "dates": dates, "windows": windows, "variants": variants,
            "returns": {t: [round(float(v), 7) for v in returns[t]] for t in returns.columns}}

    PLOTLY.parent.mkdir(parents=True, exist_ok=True)
    if not PLOTLY.exists():
        urlretrieve(f"https://cdn.plot.ly/{PLOTLY.name}", PLOTLY)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    out = DASHBOARD / "torp_triplets.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.resolve()} ({out.stat().st_size / 1e6:.2f} MB), {len(variants)} triplets × {len(windows)} windows")


if __name__ == "__main__":
    main()
