# MS thesis

Master's thesis on CNN compression via CP/Tucker decompositions — LaTeX write-up, Python/PyTorch experiments, and legacy Torch7/TensorLab code.

## Root

| File | Description |
|------|-------------|
| `MS-thesis.pdf` | Full thesis PDF (from `latex-MS-thesis/MS-thesis.pdf`) |
| `tesi-definitiva.pdf` | Presentation / defense deck (PDF slides; not a separate `.pptx`) |

## Subfolders

### `latex-source/`

Editable LaTeX: `main.tex`, `Chapters/`, `Appendices/`, `Figures/`, `bibliografia.bib`, `MastersDoctoralThesis.cls`, `compile.sh`, `setting-thesis`, `draw/`, `guide/`. Build with `./compile.sh` from this directory.

### `experiments/`

| Subfolder | Contents |
|-----------|----------|
| `papers/` | `plots.ipynb`, `thesis-plots/` (plot scripts + notebooks), `plot-scripts/` (copies of `plot_*.py` from training runs) |
| `models/` | PyTorch and Keras training code, decompositions (`cpd_workspace`, `x_decomposer`, etc.) |
| `legacy/` | CCNN, TensorLab, torchsample (Torch7-era) |
| `notes/` | `Note.txt`, `Torch Notes - 2014.txt`, `keras-thesis.code-workspace` |

Training logs, weights, and large CSV dumps were removed from the repo to keep clones small; regenerate from `experiments/models/` if needed.
