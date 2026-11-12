"""Build the classroom dataset for NEBULA LAB.

Input : the gene x cell-type feature table produced by scripts/build_dataset.py from the public
        analysis repository of Ulirsch et al., Nature Genetics 51:683-693 (2019),
        github.com/caleblareau/singlecell_bloodtraits at commit 1da2a246.
Output: data/nebula_blood_atac_rna.csv.gz, one row per gene per cell type, log2(1 + CPM) units,
        rounded to 2 decimals, written with a fixed gzip timestamp so the file is byte-reproducible.

Usage:
    python3 scripts/make_lab_data.py <features.csv.gz> data/nebula_blood_atac_rna.csv.gz
"""
import gzip
import hashlib
import io
import sys

import pandas as pd

RENAME = {
    "prom": "promoter",
    "enh10": "nearby_10kb",
    "enh50": "nearby_50kb",
    "enh100": "nearby_100kb",
    "enh250": "nearby_250kb",
    "expr": "expression",
}
# haematopoietic hierarchy, stem cells first; used for row order only
CELL_ORDER = ["HSC", "MPP", "LMPP", "CMP", "GMP-A", "GMP-B", "GMP-C", "Mono", "pDC",
              "MEP", "Ery", "CLP", "B", "CD4", "CD8", "NK"]


def main(src, out):
    df = pd.read_csv(src).rename(columns=RENAME)
    assert set(df.cell_type) == set(CELL_ORDER), sorted(set(df.cell_type) ^ set(CELL_ORDER))
    assert df.groupby("gene").size().eq(16).all(), "every gene must have all 16 cell types"
    df["cell_type"] = pd.Categorical(df.cell_type, CELL_ORDER, ordered=True)
    df = df.sort_values(["gene", "cell_type"]).reset_index(drop=True)
    df["cell_type"] = df.cell_type.astype(str)
    cols = ["gene", "cell_type", "promoter", "nearby_10kb", "nearby_50kb",
            "nearby_100kb", "nearby_250kb", "expression"]
    df = df[cols]
    for c in cols[2:]:
        df[c] = df[c].round(2)

    text = df.to_csv(index=False, float_format="%.2f").encode()
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(text)
    data = buf.getvalue()
    with open(out, "wb") as fh:
        fh.write(data)
    print(f"wrote {out}: {df.gene.nunique():,} genes x {df.cell_type.nunique()} cell types "
          f"= {len(df):,} rows, {len(data)/1e6:.2f} MB")
    print("sha256", hashlib.sha256(data).hexdigest())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
