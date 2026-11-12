# Data provenance and evaluation audit

Every number the lab computes comes from public bulk ATAC-seq and RNA-seq of 16 sorted human blood cell populations, summarised per gene by the scripts in this repository. None of it is simulated or imputed.

File: `data/nebula_blood_atac_rna.csv.gz` (3.87 MB, 292,560 rows)
SHA-256: `27e231764bc10380d5d0ff44db8606acfb0636c831a84108cf43a4123e0eef3f`
The notebook refuses to use a copy whose checksum differs.

## 1. Where the data come from

| Step | What | Checked how |
|---|---|---|
| Original measurements | Corces *et al.*, *Nat Genet* 48:1193 (2016), doi:10.1038/ng.3646: FACS-sorted human blood populations from 3 to 4 adult donors each; ATAC-seq in GEO GSE74912, RNA-seq in GSE74246. Buenrostro *et al.*, *Cell* 173:1535 (2018), doi:10.1016/j.cell.2018.03.074, sorted the GMP subpopulations; GEO SuperSeries GSE96772 (bulk ATAC GSE96771, bulk RNA GSE96811). | Accessions and descriptions confirmed on GEO mirrors (OmicsDI) and in the papers. |
| Uniform reprocessing | Ulirsch *et al.*, *Nat Genet* 51:683 (2019), doi:10.1038/s41588-019-0362-6, Online Methods. ATAC: 18 populations (16 previously reported, plus newly profiled mDCs and pDCs) processed with one pipeline, peaks resized to 500 bp around the summit, a consensus set of 451,283 peaks. RNA: raw reads realigned to hg19 with STAR 2.5.1b and summed over replicates to one count per population, for the 16 populations that match the ATAC (there is no RNA for megakaryocytes or mDCs). | Read in the full text (PubMed Central PMC6441389). |
| Files used | The paper's public analysis repository, github.com/caleblareau/singlecell_bloodtraits, commit `1da2a246` (9 July 2019): `data/bulk/ATAC/29August2017_EJCsamples_allReads_500bp.{bed,counts.txt}`, `data/bulk/RNA/16populations_RNAcounts.txt`, `data/bulk/RNA/refGene_hg19_TSS.bed`. | Downloaded and opened; 18 ATAC and 16 RNA populations, as described. |
| Per-gene summary | `scripts/build_dataset.py` (this repository) | Rebuilt twice from the downloaded files; identical output. |
| Classroom table | `scripts/make_lab_data.py`: renames columns, sorts, rounds to 2 decimals, writes gzip with a fixed timestamp | Byte-identical on rerun (same SHA-256). Rounding changes no value by more than 0.005 and moves the step 3 result from r = 0.62100 to 0.62105. |

Ulirsch *et al.* 2019 and the two source papers list where each population's samples came from; this project does not re-derive that.

## 2. What one row means

One gene in one cell type: 18,285 genes × 16 cell types = 292,560 rows, with no missing values.

| Column | Meaning | Unit |
|---|---|---|
| `gene` | RefSeq gene symbol (hg19) | |
| `cell_type` | HSC, MPP, LMPP, CMP, GMP-A, GMP-B, GMP-C, Mono, pDC, MEP, Ery, CLP, B, CD4, CD8, NK | |
| `promoter` | ATAC reads in peaks whose centre lies within 1 kb of any RefSeq NM_ start site of the gene | log₂(1 + reads per million in peaks) |
| `nearby_10kb` ... `nearby_250kb` | ATAC reads in all *other* peaks whose centre lies within 10, 50, 100 or 250 kb of a start site | same |
| `expression` | RNA-seq reads assigned to the gene | log₂(1 + reads per million) |

## 3. How the table was built

1. RNA: gene symbols that appear twice in the count table are dropped, and counts become CPM per population.
2. ATAC: counts become CPM per population, with all reads in the 451,283 consensus peaks as the library size.
3. Start sites: RefSeq NM_ transcripts (protein-coding) on chromosomes 1 to 22 and X. Genes whose symbol maps to more than one chromosome are dropped, and only genes present in the RNA table are kept.
4. For each gene, peaks are collected around every distinct start site. A peak counts once per gene, even when it sits near several start sites of that gene.
5. All features become log₂(1 + CPM).

## 4. Matching chromatin to expression

ATAC and RNA come from separately sorted bulk samples, matched by population label rather than measured in the same cells, and partly from different studies and donors. The lab therefore predicts a population's typical expression from its typical chromatin. GET's fine-tuning data were paired the same way, by cell type: pseudobulk accessibility with expression from either multiome or separate single-cell RNA-seq experiments (Fu *et al.* 2025, main text).

## 5. Evaluation design

| Question | Answer in this lab |
|---|---|
| The task | Predict each gene's expression in a cell type from its chromatin openness in that cell type. |
| The split | Hold out one whole cell type (erythroid precursors in class), train on every gene in the other 15 (274,275 rows) and test on the held-out cell type. The code asserts that no held-out row reaches training. One qualification: the 451,283 peaks were called across all 18 populations, so the held-out cell type's chromatin helped decide which regions exist, although no model ever sees its values. |
| Which genes are scored | Genes whose promoter has at least 1 ATAC read per million in the held-out cell type (log₂ value ≥ 1): 14,826 genes for erythroid precursors. The filter uses inputs only and mirrors GET's scoring of genes with accessible promoters. Optional exercise 9 shows what happens without it. |
| Preprocessing | Standardisation is fitted inside the model pipeline, on training rows only. |
| The model | Ridge regression (a straight line, α = 1). The same data always give the same answer. |
| Control | No chromatin at all: each gene's average expression over the 15 training cell types. GET's equivalent is "mean expression in training set". |
| Upper reference | Oracle: the single training cell type whose expression correlates best with the held-out answer. Because it is chosen using the answer, it is labelled an oracle and never called a model. GET's "top correlated cell type" works the same way. |
| Metrics | Pearson r (does the prediction rise and fall with the truth?); R² = 1 − SSE/SST (how close the predictions are; it can be negative); the median error as a fold change (half the genes are within it) and the root-mean-square error in log₂ units; calibration-in-the-large (the mean of measured − predicted, and the share of genes below the diagonal). Exercise 6 adds a cell-type-specific r: the correlation between the model's change for the held-out cell type (its prediction minus its own average prediction for that gene over the training cell types) and the measured change (measured minus the gene's usual level). A model whose prediction never changes between cell types scores 0. With a single input, every upward line has the same r as the input itself, so for option A it is R² that shows what was learned from other cell types. |
| No p-values | Genes are not independent samples, so the lab reports effect sizes and compares models on the same genes instead of running significance tests. |

## 6. Leakage review

| Risk | Status |
|---|---|
| Held-out cell type in training | Prevented and asserted in code. |
| Scaling fitted on test data | Prevented: the scaler is part of the pipeline and sees training rows only. |
| Gene filter using the answer | Prevented: the filter uses held-out promoter openness, an input. |
| Control using the answer | Prevented: it averages training cell types only. |
| Oracle | Uses the answer by design, and is always labelled as such. |
| Gene identity | Options A and B never see which gene a row is, so they cannot memorise genes. Exercise 6 adds each gene's usual level on purpose, and says so. |
| Related cell types | Training includes the held-out type's relatives (MEP for erythroid precursors), which cannot be avoided when one population of a hierarchy is held out; in GET's test, too, only astrocytes were held out of fine-tuning. The control stays strong without them: 0.768 without MEP and 0.766 without MEP and CMP, against 0.773 with both. It is strong because most genes sit at similar levels in every blood cell type. The oracle's best match is MEP (0.804), with a lymphoid progenitor, CLP, close behind (0.798). |

## 7. Quirks found during the audit

1. 1,163 genes (6%) have zero RNA reads in all 16 populations. Some are genes silent in blood. Others, such as LUC7L2, are broadly expressed in human tissues, so their zero more likely reflects how the source counted reads. 365 of them have an open promoter in erythroid precursors and are scored. Removing them changes r by at most 0.012 for any cell type (most for GMP-A, GMP-B and GMP-C; for erythroid precursors the model goes from 0.621 to 0.615 and the control from 0.773 to 0.768). They stay in, because removing them would mean filtering on the answer.
2. Erythroid precursors are the hardest test. 25% of their open-promoter genes have zero RNA (8 to 21% in the other populations), which is one reason the model over-predicts them on average. Their RNA library is among the smallest (32.6 M assigned reads in the count table), and haemoglobin takes about 9% of all reads. In the original study, erythroblast ATAC-seq replicates agreed far less than any other population's (technical R = 0.55, biological R = 0.50, against means of 0.94 and 0.91 for the rest; Corces *et al.* 2016), so the held-out cell type's only input is also its noisiest measurement. Their control score (0.77) is the lowest of the 16 populations (0.77 to 0.97).
3. Near-identical gene copies lose reads. HBA1 and HBA2 read 78 and 608 CPM in erythroid precursors against 79,023 for HBB, although α- and β-globin are made in matched amounts. Reads from near-identical duplicates map to several places and are discarded, an artefact of the same kind as LUC7L2.
4. CPM is not corrected for gene length. Long genes collect more reads. This mainly affects comparisons between genes, which is what Pearson r across genes measures, and it slightly favours the copy-the-average control, which inherits the same bias.
5. Peaks can be shared. A peak near two genes counts for both, so the "nearby" features of neighbouring genes overlap.

## 8. The previous lab data

The earlier version of this lab used the same public source, but packed an 8,000-gene subset (all named genes plus a random sample, seed 2026) into a 478 KB Base64/LZMA block inside the notebook, with values quantised to steps of 0.05 (ATAC) and 0.08 (RNA) log₂ units. The measurements were real and traceable, and the subset was not chosen using expression or results, but the format hid the data inside the notebook and lost precision. This version uses all 18,285 genes, rounds to 0.01, keeps the data in a separate checksummed file that loads automatically, and uses a simpler design (chromatin only; the gene's usual level appears only in exercise 6). Its numbers replace those of the previous lab.

## 9. Licences and permissions

* The source repository has no licence file. The underlying data are public GEO deposits. This repository redistributes a derived per-gene summary for non-commercial teaching, with full credit to the authors above, and will withdraw it if an author asks.
* Figure 1c of Fu *et al.* (2025) is shown unaltered (cropped from Figure 1) under CC BY-NC-ND 4.0, with credit. Published values are quoted, labelled as published, and never recomputed.

## 10. Rebuild it yourself

```bash
git clone https://github.com/caleblareau/singlecell_bloodtraits && git -C singlecell_bloodtraits checkout 1da2a246
python3 -I scripts/build_dataset.py singlecell_bloodtraits features.csv.gz
python3 scripts/make_lab_data.py features.csv.gz data/nebula_blood_atac_rna.csv.gz   # prints the SHA-256
```
