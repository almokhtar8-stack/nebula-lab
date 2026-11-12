# What step 5 quotes from the GET paper

Fu X, Mo S, Buendia A, *et al.* A foundation model of transcription across human cell types. *Nature* 637:965–973 (2025). doi:10.1038/s41586-024-08391-z. Licence: CC BY-NC-ND 4.0.

The notebook keeps two things apart and labels both. Steps 1 to 4 build your model, a simple predictor computed live on blood-cell data; it is never called GET or a reproduction of GET. Step 5 shows GET: the published Figure 1c panel, unaltered, and its printed values redrawn as bars. Nothing about GET is recomputed in this notebook.

## The figure

`assets/fu2025_fig1c.png` is the complete Figure 1c, cropped from page 2 of the published PDF (journal page 966) at 600 dpi and resized to 1,400 px wide, in full colour. No panel, label, colour or point was changed, added or removed. CC BY-NC-ND 4.0 allows sharing the material "in whole or in part" for non-commercial purposes, and resizing is a change of format, so the credit and licence line printed under the image meets the terms.

## Published values quoted (all checked in the PDF)

| Comparison on held-out fetal astrocytes | Pearson r | Where |
|---|---|---|
| GET | 0.94 | Fig. 1c panel label; main text p. 966 |
| Top correlated cell type (chosen using the answer) | 0.83 | Fig. 1c panel label |
| Mean expression in training set | 0.78 | Fig. 1c panel label |
| Gene activity | 0.51 | Fig. 1c panel label |
| TSS accessibility | 0.47 | Fig. 1c panel label |
| GET without pretraining, 100 epochs | 0.60 | Extended Data Fig. 2d bar label |
| GET without pretraining, 800 epochs | 0.61 | Extended Data Fig. 2d bar label |

The R² printed inside each Fig. 1c panel is the square of r for the panel's fitted line (0.89, 0.68, 0.60, 0.22, 0.26). The main text reports R² as agreement between prediction and measurement (0.88, 0.62, 0.53, −0.23, −0.67). The notebook uses the main-text definition, and the caption under the figure says the panel values cannot be compared with it.

## How the setting differs from the lab

| | Lab (step 3) | Fig. 1c |
|---|---|---|
| Cells | 16 sorted human blood populations, bulk | 213 fetal and adult cell types, single-cell pseudobulk |
| Held out | Erythroid precursors, from all training | Fetal astrocytes, from expression fine-tuning only; their chromatin was in pretraining |
| Genes scored | Genes with an open promoter (14,826) | Genes with an accessible promoter |
| Expression unit | log₂(1 + CPM) | log(TPM) as labelled on the axes |
| Model | Ridge regression on 1 or 2 numbers | 12-layer transformer on 200 regions × 283 features |

Step 5 therefore compares only the order of the bars. In both experiments accessibility alone loses to copying the average, and GET is the only real predictor that beats it. The top correlated cell type beats it too, but it is picked using the answer. Above each published bar, step 5 prints the paper's own panel label in quotation marks.

## Why Figure 1c is shown and not reproduced

A reproduction from the authors' released numbers would have been better, but none of the routes is feasible for a class.

1. Source data: the article has no Source Data files.
2. Figure code archive: Zenodo 10.5281/zenodo.13357634 (CC BY 4.0) is a single `get_figure_code.tar.gz` of 41,423,663,246 bytes. A download could stop once the Figure 1 files are reached, but their position in the archive is unknown, so the worst case is the full 41 GB.
3. Model checkpoint: the leave-out-astrocyte checkpoint used by the authors' tutorial sits in the S3 bucket `2023-get-xf2217`, which now refuses anonymous requests (Requester Pays). The 79 MB astrocyte input (`astrocyte.zarr`, Zenodo 14614947, CC BY 4.0) is public, but GET's predictions cannot be regenerated without the checkpoint.

Step 5 therefore shows the original panel, labelled as published.

One route has not been tried yet. The TSS-accessibility panel (r = 0.47) needs no model, only accessibility and expression, so a Colab session could read `astrocyte.zarr` and test whether it reproduces that panel. It would be added to the notebook only if it matched the published value within rounding.
