"""Run a notebook's code cells as a plain script, without writing outputs back into the notebook.

Usage (from anywhere): python run_notebook.py <notebook.ipynb> <figure folder>

Cells run in order in one namespace, with the notebook's folder as working directory, so relative
paths behave as in Jupyter. Figures passed to plt.show() are saved as <figure folder>/fig_NN.png;
display() prints tables. Long unattended runs use this instead of nbconvert so the notebooks stay
output-free and a crash leaves the per-run caches intact for a resume.

Everything runs under the `__main__` guard: on Windows, multiprocessing workers (e.g. in
`jepa_experiments.features.rqa_features`) re-import this file and must not execute the notebook again.
"""
import json
import os
import sys
from pathlib import Path


def main(notebook: Path, figure_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import IPython.display

    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_count = 0

    def save_figures(*args, **kwargs):
        nonlocal figure_count
        for number in plt.get_fignums():
            figure_count += 1
            plt.figure(number).savefig(figure_dir / f"fig_{figure_count:02d}.png", dpi=120, bbox_inches="tight")
        plt.close("all")

    def print_display(*objects, **kwargs):
        for obj in objects:
            print(obj.to_string() if hasattr(obj, "to_string") else obj, flush=True)

    plt.show = save_figures
    IPython.display.display = print_display
    os.chdir(notebook.parent)
    namespace = {"__name__": "__main__", "display": print_display}  # Jupyter provides display() as a builtin
    for index, cell in enumerate(json.loads(notebook.read_text(encoding="utf-8"))["cells"]):
        if cell["cell_type"] == "code":
            print(f"\n# ---- cell {index} ----", flush=True)
            exec(compile("".join(cell["source"]), f"{notebook.name}:cell{index}", "exec"), namespace)
    print(f"\nDone: {notebook.name}; {figure_count} figures in {figure_dir}", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
