#!/usr/bin/env python3
"""Build the NEBULA mini-lab dataset from public bulk ATAC-seq + RNA-seq of
sorted human blood cell populations.

Source (real, measured data; nothing simulated):
  GitHub caleblareau/singlecell_bloodtraits (analysis repository of
  Ulirsch, Lareau, Bao et al., Nature Genetics 51:683-693, 2019)
    data/bulk/RNA/16populations_RNAcounts.txt   bulk RNA-seq counts, 16 populations
    data/bulk/RNA/refGene_hg19_TSS.bed          RefSeq TSS coordinates (hg19)
    data/bulk/ATAC/29August2017_EJCsamples_allReads_500bp.bed          500 bp peaks
    data/bulk/ATAC/29August2017_EJCsamples_allReads_500bp.counts.txt   ATAC counts, 18 populations

Output (one row per gene x cell type; 16 cell types present in both assays):
  gene, cell_type, prom, enh10, enh50, enh100, enh250, expr
  prom      log2(1 + ATAC CPM in peaks whose centre is within 1 kb of any RefSeq NM_ TSS of the gene)
  enhW      log2(1 + ATAC CPM in other peaks whose centre is within W kb of a TSS (W = 10, 50, 100, 250))
  expr      log2(1 + RNA CPM)
The table is the only input the notebook needs. Usage:
  python3 -I build_dataset.py <path/to/singlecell_bloodtraits> <out.csv.gz>
"""
import sys
import numpy as np
import pandas as pd

WINDOWS_KB = (10, 50, 100, 250)
PROM_BP = 1000


def main(src, out):
    rna = pd.read_csv(f"{src}/data/bulk/RNA/16populations_RNAcounts.txt", sep="\t")
    rna = rna[~rna["Genes"].duplicated(keep=False)].set_index("Genes")
    cell_types = list(rna.columns)

    atac = pd.read_csv(f"{src}/data/bulk/ATAC/29August2017_EJCsamples_allReads_500bp.counts.txt", sep="\t")
    peaks = pd.read_csv(f"{src}/data/bulk/ATAC/29August2017_EJCsamples_allReads_500bp.bed",
                        sep="\t", header=None, names=["chr", "start", "end"])
    assert len(atac) == len(peaks)
    missing = [c for c in cell_types if c not in atac.columns]
    assert not missing, missing
    # CPM within the full peak set, per cell type (library size = all reads in peaks)
    atac_cpm = atac[cell_types].div(atac[cell_types].sum(axis=0), axis=1) * 1e6
    rna_cpm = rna.div(rna.sum(axis=0), axis=1) * 1e6

    tss = pd.read_csv(f"{src}/data/bulk/RNA/refGene_hg19_TSS.bed", sep="\t", header=None,
                      names=["chr", "start", "end", "acc", "gene", "strand"])
    tss = tss[tss.acc.str.startswith("NM_") & tss.chr.str.match(r"^chr([0-9]+|X)$")]
    tss = tss[tss.gene.isin(rna_cpm.index)]
    # a gene symbol must map to a single chromosome
    one_chr = tss.groupby("gene").chr.nunique()
    tss = tss[tss.gene.isin(one_chr[one_chr == 1].index)]

    centers = ((peaks.start + peaks.end) // 2).to_numpy()
    by_chr = {}
    for ch, idx in peaks.groupby("chr").indices.items():
        order = np.argsort(centers[idx])
        by_chr[ch] = (idx[order], centers[idx][order])

    cpm = atac_cpm.to_numpy()
    rows = []
    for gene, g in tss.groupby("gene"):
        ch = g.chr.iloc[0]
        if ch not in by_chr:
            continue
        pidx, pc = by_chr[ch]
        sites = np.unique(g.start.to_numpy())

        def peaks_within(bp):
            sel = set()
            for s in sites:
                lo, hi = np.searchsorted(pc, s - bp, "left"), np.searchsorted(pc, s + bp, "right")
                sel.update(pidx[lo:hi].tolist())
            return sel

        prom = peaks_within(PROM_BP)
        feats = {"prom": cpm[list(prom)].sum(axis=0) if prom else np.zeros(len(cell_types))}
        for w in WINDOWS_KB:
            distal = peaks_within(w * 1000) - prom
            feats[f"enh{w}"] = cpm[list(distal)].sum(axis=0) if distal else np.zeros(len(cell_types))
        expr = rna_cpm.loc[gene].to_numpy()
        for j, c in enumerate(cell_types):
            rows.append([gene, c] + [feats[k][j] for k in ["prom"] + [f"enh{w}" for w in WINDOWS_KB]] + [expr[j]])

    cols = ["gene", "cell_type", "prom"] + [f"enh{w}" for w in WINDOWS_KB] + ["expr"]
    df = pd.DataFrame(rows, columns=cols)
    for c in cols[2:]:
        df[c] = np.log2(1 + df[c]).round(4)
    df.to_csv(out, index=False, compression="gzip")
    print(f"wrote {out}: {df.gene.nunique()} genes x {df.cell_type.nunique()} cell types = {len(df)} rows")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
