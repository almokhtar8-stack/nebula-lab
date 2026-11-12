"""NEBULA LAB: the helper code behind the classroom notebook.

You never need to read or edit this file to do the lab. It loads the data, draws the figures
and keeps the bookkeeping out of the notebook, so that each step stays a few lines long.

Everything it reports is computed when you run it, from real measurements. The only numbers
written into this file are the published values from Fu et al. (2025) that step 5 quotes, and
they are labelled as published values wherever they appear.
"""
from __future__ import annotations

import hashlib
import html
import os
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from IPython.display import HTML, display
except ImportError:                       # plain Python: print text instead of HTML
    HTML = None

    def display(obj):
        print(obj)

# --------------------------------------------------------------------------------------------
# where the data live (a fixed release, so every student gets identical files)
# --------------------------------------------------------------------------------------------
REPO = "almokhtar8-stack/nebula-lab"
REF = "v1.0"
DATA_FILE = "data/nebula_blood_atac_rna.csv.gz"
DATA_SHA256 = "27e231764bc10380d5d0ff44db8606acfb0636c831a84108cf43a4123e0eef3f"
MIRRORS = (
    "https://raw.githubusercontent.com/{repo}/{ref}/{path}",
    "https://cdn.jsdelivr.net/gh/{repo}@{ref}/{path}",
)
REPO_URL = f"https://github.com/{REPO}"
COMPLETED_URL = f"https://github.com/{REPO}/blob/{REF}/NEBULA_Lab_completed.ipynb"

# --------------------------------------------------------------------------------------------
# design: one colour per kind of thing, used the same way in every figure
# --------------------------------------------------------------------------------------------
BG = "#FAFCF8"        # warm white page
INK = "#123E38"       # dark evergreen: text, and measured expression
TEAL = "#17645B"      # chromatin openness, the model's input
RUST = "#C45B39"      # the model's predictions
GREY = "#8C9A94"      # the no-chromatin control
GRID = "#E3EAE5"
MUTED = "#5E7A72"
PALE_TEAL = "#CFE3DD"
PALE_RUST = "#F1D3C7"
ORACLE = "#B9C4C0"    # the best-matching training cell type (uses the answer)
GET_LIGHT = "#8FB9AF" # GET variants without pretraining

CELL_NAMES = {
    "HSC": "haematopoietic stem cells", "MPP": "multipotent progenitors",
    "LMPP": "lymphoid-primed multipotent progenitors", "CMP": "common myeloid progenitors",
    "GMP-A": "granulocyte-macrophage progenitors (A)", "GMP-B": "granulocyte-macrophage progenitors (B)",
    "GMP-C": "granulocyte-macrophage progenitors (C)", "Mono": "monocytes",
    "pDC": "plasmacytoid dendritic cells", "MEP": "megakaryocyte-erythroid progenitors",
    "Ery": "erythroid precursors", "CLP": "common lymphoid progenitors", "B": "B cells",
    "CD4": "CD4 T cells", "CD8": "CD8 T cells", "NK": "natural killer cells",
}
CELL_ORDER = list(CELL_NAMES)
# marker genes labelled on the figures, per held-out cell type (only those present are drawn)
MARKERS = {
    "Ery": ["HBB", "GYPA", "ALAS2", "SLC4A1"], "MEP": ["GATA1", "KLF1", "TFRC", "ITGA2B"],
    "B": ["PAX5", "MS4A1", "CD79A", "CD19"], "Mono": ["LYZ", "CD14", "S100A8", "FCN1"],
    "NK": ["GNLY", "NKG7", "KLRD1", "PRF1"], "CD4": ["CD4", "IL7R", "CCR7", "CD3E"],
    "CD8": ["CD8A", "CD8B", "GZMK", "CD3E"], "pDC": ["IRF8", "LILRA4", "IL3RA", "TCF4"],
    "CLP": ["DNTT", "IL7R", "CD34", "VPREB1"], "HSC": ["CD34", "HLF", "CRHBP", "PROM1"],
    "MPP": ["CD34", "HLF", "PROM1", "SPINK2"], "LMPP": ["CD34", "FLT3", "SPINK2", "PROM1"],
    "CMP": ["CD34", "GATA2", "MPO", "KIT"], "GMP-A": ["MPO", "ELANE", "AZU1", "CTSG"],
    "GMP-B": ["MPO", "ELANE", "AZU1", "CTSG"], "GMP-C": ["MPO", "ELANE", "LYZ", "CTSG"],
}
HEATMAP_GENES = ["HBB", "GYPA", "PAX5", "MS4A1", "CD3E", "GNLY", "LYZ", "CD34", "ACTB"]

FEATURE_SETS = {
    "Option A: promoter features only": ["promoter"],
    "Option B: promoter + nearby regulatory features": ["promoter", "nearby_50kb"],
}
FEATURE_NAMES = {
    "promoter": "promoter openness", "nearby_10kb": "nearby openness (within 10 kb)",
    "nearby_50kb": "nearby openness (within 50 kb)", "nearby_100kb": "nearby openness (within 100 kb)",
    "nearby_250kb": "nearby openness (within 250 kb)",
}
OPEN = 1.0   # genes are scored when their promoter has at least 1 ATAC read per million (log2 1+CPM >= 1)

# Published values, Fu et al., Nature 637:965-973 (2025), Figure 1c (fetal astrocytes held out of
# expression fine-tuning). Keys are the panel labels as printed; values are Pearson r as printed.
# Quoted, not recomputed.
PUBLISHED_FIG1C = {
    "TSS accessibility": 0.47,
    "Gene activity": 0.51,
    "Mean expression in training set": 0.78,
    "Top correlated cell type": 0.83,
    "GET prediction": 0.94,
}
PUBLISHED_NO_PRETRAINING = {"GET, pretrained then fine-tuned": 0.94,
                            "Same model, no pretraining (100 epochs)": 0.60,
                            "Same model, no pretraining (800 epochs)": 0.61}
CITATION = ("Fu X, Mo S, Buendia A, et al. A foundation model of transcription across human cell "
            "types. Nature 637:965-973 (2025). doi:10.1038/s41586-024-08391-z")

_STATE = {"results": {}}


# --------------------------------------------------------------------------------------------
# look and feel
# --------------------------------------------------------------------------------------------
def _style():
    matplotlib.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": "#B7C6C0", "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
        "axes.titleweight": "bold", "axes.labelsize": 11, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "figure.dpi": 100, "axes.axisbelow": True,
    })


_style()


def _html(s):
    if HTML is None:
        import re
        print(re.sub(r"<[^>]+>", "", s))
    else:
        display(HTML(s))


def _show(fig):
    """Display a figure at twice the screen resolution, labelled with its width only. Colab sizes
    an image from both width and height when both are given, so a screen narrower than the figure
    squeezes it sideways; with the width alone it scales down evenly."""
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=2 * fig.dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor(), edgecolor=fig.get_edgecolor())
    plt.close(fig)
    if HTML is None:                      # plain Python: nothing to draw on
        return
    from IPython.display import Image
    png = buf.getvalue()
    width = int.from_bytes(png[16:20], "big") // 2          # PNG header: pixel width
    display(Image(data=png, format="png", width=width))


def _card(title, body, kind="info"):
    rule = {"info": TEAL, "done": INK, "warn": RUST, "control": GREY}[kind]
    return (f'<div style="background:{BG};color:{INK};font-family:sans-serif;max-width:860px;'
            f'margin:12px 0;padding:9px 12px 10px;border-top:2px solid {rule};">'
            f'<div style="font-weight:700;font-size:14.5px;margin-bottom:3px;">{title}</div>'
            f'<div style="font-size:14px;line-height:1.55;">{body}</div></div>')


def _note(text):
    """One short paragraph under a result. Printed text never wraps in Colab, so long sentences
    go here instead."""
    _html(f'<div style="color:{INK};background:{BG};font-family:sans-serif;font-size:14px;'
          f'line-height:1.5;max-width:860px;padding:6px 12px;">{text}</div>')


def _scroll(inner):
    """Wrap a table so that a narrow screen scrolls it sideways instead of cutting columns off."""
    return (f'<div style="overflow-x:auto;max-width:100%;background:{BG};padding:8px 12px;'
            f'border-radius:6px;">{inner}</div>')


def _title(fig, text, sub=None, gap=0.42):
    """Figure title and subtitle, wrapped to the figure's width. Returns the `top` fraction the
    axes should start at, so titles never collide with plots whatever the figure size."""
    import textwrap
    w_in, h_in = fig.get_size_inches()
    t_lines = textwrap.wrap(text, width=max(30, int(w_in * 7.3)))
    s_lines = textwrap.wrap(sub, width=max(40, int(w_in * 11.0))) if sub else []
    fig.text(0.012, 1 - 0.10 / h_in, "\n".join(t_lines), ha="left", va="top", fontsize=15,
             fontweight="bold", family="DejaVu Serif", color=INK, linespacing=1.15)
    used = 0.10 + 0.27 * len(t_lines)
    if s_lines:
        fig.text(0.012, 1 - (used + 0.06) / h_in, "\n".join(s_lines), ha="left", va="top",
                 fontsize=11, color=MUTED, linespacing=1.3)
        used += 0.06 + 0.21 * len(s_lines)
    return 1 - (used + gap) / h_in


def _metrics(measured, predicted):
    """Pearson r; R^2 = 1 - SSE/SST (how close, can be negative); root-mean-square and median
    errors in log2 units; calibration-in-the-large (mean of measured - predicted)."""
    y, p = np.asarray(measured, float), np.asarray(predicted, float)
    err = y - p
    return {"r": float(np.corrcoef(y, p)[0, 1]),
            "R2": float(1 - (err ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "med_fold": float(2 ** np.median(np.abs(err))),
            "bias": float(err.mean()),
            "below": float((y < p).mean()),
            "n": int(len(y))}


def _equation(model, features):
    scaler, ridge = model[0], model[-1]
    slopes = ridge.coef_ / scaler.scale_
    intercept = ridge.intercept_ - float((ridge.coef_ * scaler.mean_ / scaler.scale_).sum())
    return dict(zip(features, slopes)), float(intercept)


def _minus(x, fmt="+.3f"):
    return format(x, fmt).replace("-", "&minus;")


# --------------------------------------------------------------------------------------------
# step 1: data
# --------------------------------------------------------------------------------------------
def _fetch(path, sha256=None):
    """Return a local path to `path`, downloading it from GitHub (or a mirror) if needed."""
    if os.path.exists(path):
        with open(path, "rb") as fh:
            if sha256 is None or hashlib.sha256(fh.read()).hexdigest() == sha256:
                return path
        os.remove(path)                     # a damaged or outdated copy: fetch a fresh one
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tried = []
    for template in MIRRORS:
        url = template.format(repo=REPO, ref=REF, path=path)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                blob = resp.read()
            if sha256 and hashlib.sha256(blob).hexdigest() != sha256:
                tried.append(f"{url}: file differs from the verified version")
                continue
            with open(path, "wb") as fh:
                fh.write(blob)
            return path
        except Exception as exc:        # network down, blocked host, timeout
            tried.append(f"{url}: {type(exc).__name__}")
    _html(_card("The data could not be downloaded",
                "The network blocked the download. Follow along on the projector, or see every result on "
                f'<a href="{REPO_URL}" target="_blank">the project page</a>.', "warn"))
    raise SystemExit("Data download failed. Tried: " + "; ".join(tried))


def load_data():
    """Load the 18,285-gene x 16-cell-type table and check it."""
    df = pd.read_csv(_fetch(DATA_FILE, DATA_SHA256))
    expected = ["gene", "cell_type", "promoter", "nearby_10kb", "nearby_50kb",
                "nearby_100kb", "nearby_250kb", "expression"]
    assert list(df.columns) == expected, f"unexpected columns: {list(df.columns)}"
    assert set(df.cell_type) == set(CELL_ORDER), "unexpected cell types"
    assert not df.isna().any().any(), "missing values in the table"
    _STATE["data"] = df
    return df


def show_data(df):
    """Step 1 output: what one row means, a few example rows, and a marker-gene picture."""
    n_genes, n_types = df.gene.nunique(), df.cell_type.nunique()
    _html(_card("Data loaded",
                f"<b>{n_genes:,} genes &times; {n_types} blood cell types = {len(df):,} measurements.</b> "
                "Each row is one gene in one cell type: how open its DNA is (ATAC-seq) and how much "
                "RNA it makes (RNA-seq). Values are log&#8322;(1 + CPM), where CPM is counts per million "
                "reads, so a step of 1 is roughly a doubling.", "done"))
    picks = [("HBB", "Ery"), ("HBB", "B"), ("PAX5", "B"), ("PAX5", "Ery"), ("ACTB", "Ery"), ("ACTB", "B")]
    rows = pd.concat([df[(df.gene == g) & (df.cell_type == c)] for g, c in picks])
    th = (f'style="text-align:left;padding:6px 14px;border-bottom:2px solid {INK};'
          f'color:{INK};font-weight:700;"')
    td = f'style="padding:5px 14px;border-bottom:1px solid {GRID};color:{INK};text-align:left;"'
    tn = f'style="padding:5px 14px;border-bottom:1px solid {GRID};color:{INK};text-align:right;"'
    head = "".join(f"<th {th}>{h}</th>" for h in
                   ["Gene", "Cell type", "Promoter openness", "Nearby openness (50 kb)", "Measured expression"])
    body = "".join(
        f"<tr><td {td}><b>{r.gene}</b></td><td {td}>{CELL_NAMES[r.cell_type]}</td>"
        f"<td {tn}>{r.promoter:.2f}</td><td {tn}>{r.nearby_50kb:.2f}</td><td {tn}><b>{r.expression:.2f}</b></td></tr>"
        for r in rows.itertuples())
    _html(_scroll(
        f'<table style="border-collapse:collapse;font-family:sans-serif;font-size:14px;'
        f'background:{BG};margin:2px 0 4px;white-space:nowrap;">{head}{body}</table>'
        f'<div style="color:{MUTED};font-size:13px;font-family:sans-serif;max-width:820px;">'
        "Six of the 292,560 rows. HBB, the &beta;-globin gene of adult haemoglobin, is wide open and highly "
        "expressed in erythroid precursors but barely open and nearly silent in B cells. PAX5 is the "
        "reverse, and ACTB is open and on everywhere.</div>"))
    _plot_markers(df)


def _plot_markers(df):
    genes = [g for g in HEATMAP_GENES if g in set(df.gene)]
    sub = df[df.gene.isin(genes)]
    access = sub.pivot(index="gene", columns="cell_type", values="promoter").loc[genes, CELL_ORDER]
    expr = sub.pivot(index="gene", columns="cell_type", values="expression").loc[genes, CELL_ORDER]
    access = access.div(access.max(axis=1), axis=0)      # each gene shaded against its own maximum
    expr = expr.div(expr.max(axis=1), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"wspace": 0.08})
    cmaps = [matplotlib.colors.LinearSegmentedColormap.from_list("t", [BG, PALE_TEAL, TEAL]),
             matplotlib.colors.LinearSegmentedColormap.from_list("i", [BG, "#9DB9B1", INK])]
    for ax, mat, cmap, label in zip(axes, [access, expr], cmaps,
                                    ["Promoter openness (ATAC-seq)", "Measured expression (RNA-seq)"]):
        ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(CELL_ORDER)))
        ax.set_xticklabels(CELL_ORDER, rotation=90, fontsize=9.5)
        ax.set_yticks(range(len(genes)))
        ax.set_yticklabels(genes if ax is axes[0] else [], fontsize=10.5)
        ax.grid(False)
        ax.set_title(label, fontsize=12, color=TEAL if ax is axes[0] else INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    top = _title(fig, "Same genes, same 16 cell types: openness and expression mostly rise together",
                 "Each row is shaded against that gene's own maximum: white is none, darkest is where the gene "
                 "is most open or most expressed. The two mostly match, but open DNA does not always mean an "
                 "expressed gene.")
    fig.subplots_adjust(top=top, bottom=0.17, left=0.06, right=0.99)
    _show(fig)


# --------------------------------------------------------------------------------------------
# step 2: train
# --------------------------------------------------------------------------------------------
def _split(df, held_out):
    if held_out not in CELL_ORDER:
        raise ValueError(f"held_out must be one of {CELL_ORDER}")
    train = df[df.cell_type != held_out].reset_index(drop=True)
    test = df[(df.cell_type == held_out) & (df.promoter >= OPEN)].reset_index(drop=True)
    # the no-chromatin control: each gene's average expression in the 15 training cell types
    test["average_in_training"] = test.gene.map(train.groupby("gene").expression.mean())
    assert held_out not in set(train.cell_type)                      # nothing leaks into training
    assert test.cell_type.eq(held_out).all()
    return train, test


def split(df, held_out="Ery"):
    """Training set: every gene in the other 15 cell types. Test set: the held-out cell type,
    scored on genes whose promoter is open there (an input-only filter, as GET does)."""
    train, test = _split(df, held_out)
    _STATE.update(held_out=held_out, results={})
    return train, test


def make_model():
    """A straight-line model: standardise the inputs (using training data only), then ridge regression."""
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


def show_training(model, train, features=("promoter",)):
    """Step 2 output: confirm what was learned, from how much data, and draw it."""
    features = list(features)
    slopes, intercept = _equation(model, features)
    n_types = train.cell_type.nunique()
    terms = " + ".join(f"{v:.2f} &times; {FEATURE_NAMES[k]}" for k, v in slopes.items())
    _html(_card("Trained",
                f"The model learned from <b>{len(train):,} examples</b> "
                f"({train.gene.nunique():,} genes in {n_types} cell types), none of them from "
                f"{CELL_NAMES[_STATE['held_out']]}.<br>"
                f"The fitted line: <b>predicted expression = {terms} {'+' if intercept >= 0 else '&minus;'} "
                f"{abs(intercept):.2f}</b>", "done"))
    if features != ["promoter"]:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    hb = ax.hexbin(train.promoter, train.expression, gridsize=55, mincnt=1, bins="log",
                   cmap=matplotlib.colors.LinearSegmentedColormap.from_list("h", [PALE_TEAL, TEAL, INK]),
                   linewidths=0)
    xs = np.linspace(0, train.promoter.max(), 50)
    ax.plot(xs, model.predict(pd.DataFrame({"promoter": xs})), color=RUST, lw=3,
            label="what the model learned")
    ax.set_xlabel("Promoter openness, log$_2$(ATAC CPM + 1)")
    ax.set_ylabel("Measured expression, log$_2$(RNA CPM + 1)")
    ax.legend(loc="upper left", fontsize=11)
    cb = fig.colorbar(hb, ax=ax, pad=0.01)
    cb.set_label("training examples per hexagon", color=MUTED)
    cb.outline.set_visible(False)
    top = _title(fig, f"Training: one line through {len(train):,} examples",
                 "Genes with more open promoters tend to make more RNA. The spread around the line "
                 "is what the model cannot explain.", gap=0.2)
    fig.subplots_adjust(top=top, left=0.1, right=0.98, bottom=0.12)
    _show(fig)


# --------------------------------------------------------------------------------------------
# step 3: predict
# --------------------------------------------------------------------------------------------
def _label_points(ax, xs, ys, names, color):
    """Label a few points in a column to their right, with leader lines, so labels never overlap."""
    span_y = ax.get_ylim()[1] - ax.get_ylim()[0]
    span_x = ax.get_xlim()[1] - ax.get_xlim()[0]
    xcol = float(np.max(xs)) + 0.09 * span_x
    last = None
    for i in np.argsort(ys)[::-1]:
        y = float(ys[i]) if last is None else min(float(ys[i]), last - 0.06 * span_y)
        last = y
        ax.annotate(names[i], (xs[i], ys[i]), xytext=(xcol, y), fontsize=10.5, color=color,
                    fontweight="bold", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8, shrinkA=0, shrinkB=3))


def _scatter(ax, test, predicted, title, metrics, color=RUST, markers=True, decimals=2):
    y = test.expression.to_numpy()
    ax.scatter(predicted, y, s=5, alpha=0.22, color=color, linewidths=0, rasterized=True)
    lo, hi = 0, max(float(np.max(y)), float(np.max(predicted))) + 0.5
    ax.plot([lo, hi], [lo, hi], ls="--", color=INK, lw=1.2, alpha=0.7)
    ax.text(hi * 0.98, hi * 0.80, "perfect\nprediction", ha="right", va="top", fontsize=10, color=INK)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted expression, log$_2$(RNA CPM + 1)")
    ax.set_ylabel("Measured expression, log$_2$(RNA CPM + 1)")
    ax.set_title(title, loc="left", fontsize=12.5)
    ax.text(0.03, 0.97,
            f"r = {metrics['r']:.{decimals}f}\nR$^2$ = {metrics['R2']:.2f}\n"
            f"median error: {metrics['med_fold']:.1f}-fold",
            transform=ax.transAxes, ha="left", va="top", fontsize=12, color=INK,
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=GRID))
    if markers:
        names = [g for g in MARKERS.get(_STATE.get("held_out"), []) if g in set(test.gene)]
        idx = [int(np.flatnonzero(test.gene.to_numpy() == g)[0]) for g in names]
        if idx:
            px, py = np.asarray(predicted)[idx], y[idx]
            ax.scatter(px, py, s=46, color=INK, edgecolor="white", linewidth=1.2, zorder=5)
            _label_points(ax, px, py, names, INK)


def show_predictions(test, predicted):
    """Step 3 output: predicted against measured expression for the held-out cell type."""
    predicted = np.asarray(predicted, float)
    m = _metrics(test.expression, predicted)
    _STATE["results"]["A"] = m
    _STATE["pred_A"] = predicted
    held = _STATE["held_out"]
    fig, ax = plt.subplots(figsize=(7.4, 7.3))
    _scatter(ax, test, predicted, "Promoter openness only", m)
    top = _title(fig, f"Predictions for {CELL_NAMES[held]}, a cell type the model never saw",
                 f"Each dot is one gene ({m['n']:,} genes with an open promoter). "
                 f"Labelled: marker genes of {CELL_NAMES[held]}.")
    fig.subplots_adjust(top=top, left=0.12, right=0.97, bottom=0.09)
    _show(fig)

    err = test.expression.to_numpy() - predicted
    worst = test.gene.to_numpy()[int(np.argmax(np.abs(err)))]
    df, open_train = _STATE.get("data"), None
    if df is not None:
        tr = df[(df.cell_type != held) & (df.promoter >= OPEN)]
        open_train = float((tr.expression == 0).mean())
    silent_here = float((test.expression == 0).mean())
    if m["bias"] < -0.2:
        why = (f" One reason: in {CELL_NAMES[held]}, {silent_here:.0%} of genes with an open promoter make no "
               f"RNA at all, against {open_train:.0%} in the training cell types." if open_train is not None else "")
        cal = (f"<b>{m['below']:.0%} of genes sit below the dashed line</b>: the model guesses too high, by "
               f"{-m['bias']:.2f} log&#8322; units on average.{why}")
    elif m["bias"] > 0.2:
        cal = (f"<b>{1 - m['below']:.0%} of genes sit above the dashed line</b>: the model guesses too low, by "
               f"{m['bias']:.2f} log&#8322; units on average.")
    else:
        cal = ("The dots sit on both sides of the dashed line: on average the model is off by only "
               f"{_minus(m['bias'], '+.2f')} log&#8322; units.")
    close = ("they are still far from the measured values" if m["R2"] < 0.5
             else "they are close to the measured values")
    _html(_card("Reading the plot",
                f"<b>r = {m['r']:.2f}</b>: the predictions rise and fall with the measurements "
                "(1 would be perfect, 0 no relationship).<br>"
                f"<b>R&sup2; = {m['R2']:.2f}</b>: {close}. The median error is {m['med_fold']:.1f}-fold, "
                f"and {worst} is the worst miss.<br>"
                f"{cal}<br>"
                f'<span style="color:{MUTED};">With a single input, any rising line gives the same r, so R&sup2; '
                "is the fairer test of what the model learned from the other 15 cell types. A line fitted to "
                f"these very genes would reach R&sup2; = {m['r'] ** 2:.2f}.</span>"))


# --------------------------------------------------------------------------------------------
# step 4: experiment
# --------------------------------------------------------------------------------------------
def compare(test, predicted_before, predicted_now, features, model_now):
    """Step 4 output: the step-3 model and the model just trained, side by side, against the
    no-chromatin control."""
    features = list(features)
    key = "B" if "nearby_50kb" in features else "A"
    m_before = _STATE["results"].get("A") or _metrics(test.expression, predicted_before)
    m_now = _metrics(test.expression, predicted_now)
    _STATE["results"][key] = m_now
    m_ctrl = _metrics(test.expression, test.average_in_training)
    _STATE["results"]["control"] = m_ctrl
    slopes, _ = _equation(model_now, features)
    held = _STATE["held_out"]

    fig = plt.figure(figsize=(12.5, 8.8))
    top = _title(fig, f"Experiment: does nearby regulatory DNA improve the prediction for {CELL_NAMES[held]}?",
                 "Same held-out genes, same kind of model; only the input changes. Grey: a control that "
                 "ignores chromatin entirely.")
    bars = [("A: promoter features only", m_before["r"], PALE_RUST)]
    if key == "B":
        bars.append(("B: promoter + nearby features (50 kb)", m_now["r"], RUST))
    bars.append(("Control: copy each gene's average\n(no chromatin at all)", m_ctrl["r"], GREY))
    bar_h = 0.065 * len(bars) + 0.04
    axb = fig.add_axes([0.30, top - bar_h, 0.66, bar_h])
    yy = np.arange(len(bars))[::-1]
    for yv, (label, v, col) in zip(yy, bars):
        axb.barh(yv, v, color=col, height=0.66)
        axb.text(v + 0.012, yv, f"r = {v:.3f}", va="center", fontsize=12.5, fontweight="bold", color=INK)
    axb.set_yticks(yy)
    axb.set_yticklabels([b[0] for b in bars], fontsize=11.5, color=INK)
    axb.set_xlim(0, 1.1)
    axb.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axb.set_xlabel("Pearson r on the same held-out genes", fontsize=10.5)
    axb.grid(axis="y", visible=False)
    axb.tick_params(axis="y", length=0)
    gs = gridspec.GridSpec(1, 2, wspace=0.22, left=0.07, right=0.98, top=top - bar_h - 0.115, bottom=0.07)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    _scatter(ax1, test, predicted_before, "Option A (step 3): promoter features only", m_before,
             markers=False, decimals=3)
    now_title = ("Your choice, option B: promoter + nearby features" if key == "B"
                 else "Your choice, option A: the step 3 model again")
    _scatter(ax2, test, predicted_now, now_title, m_now, markers=False, decimals=3)
    _show(fig)

    if key == "B":
        diff = m_now["r"] - m_before["r"]
        if abs(diff) < 0.01:
            verdict = "made <b>no meaningful difference</b>"
            why = " Summed openness within 50 kb added almost nothing about how much RNA a gene makes."
        elif diff > 0:
            verdict, why = "<b>improved</b> the prediction", ""
        else:
            verdict, why = "made the prediction <b>worse</b>", ""
        _html(_card("Result",
                    f"Adding nearby regions {verdict}: r went from {m_before['r']:.3f} to {m_now['r']:.3f} "
                    f"(a change of {_minus(diff)}). The model gave nearby openness a weight of "
                    f"<b>{slopes['nearby_50kb']:.2f}</b>, against <b>{slopes['promoter']:.2f}</b> for the "
                    f"promoter.{why} Optional exercise 3 runs the same test for all 16 cell types.", "done"))
    else:
        _html(_card("Same model, same answer",
                    "Option A trains exactly the step 3 model, so the result is identical: the analysis is "
                    "deterministic. Choose option B and run the cell again to test nearby regions.", "info"))
    best = max(m_before["r"], m_now["r"])
    if m_ctrl["r"] > best:
        ctrl_text = (f"reaches <b>r = {m_ctrl['r']:.2f}</b>, higher than either accessibility model. Most genes "
                     "sit at similar levels in every blood cell type, so knowing <i>which gene</i> it is matters "
                     "more than how open its DNA is.")
    else:
        ctrl_text = (f"reaches <b>r = {m_ctrl['r']:.2f}</b>, lower than the accessibility model here: in this "
                     "cell type, chromatin adds something the average cannot.")
    _html(_card("The grey bar",
                "Copying each gene's average expression from the 15 training cell types, without looking at "
                f"chromatin, {ctrl_text} GET had to beat the same control.", "control"))
    _html(_card("Two cautions",
                "Adding inputs does not guarantee a better model. And if B had won, that would show only that "
                "nearby openness is <i>associated</i> with expression. Showing that those regions "
                "<i>regulate</i> the gene takes an experiment, such as silencing the region and measuring the "
                "RNA.", "warn"))


# --------------------------------------------------------------------------------------------
# step 5: GET
# --------------------------------------------------------------------------------------------
def meet_get():
    """Step 5 output: your results next to the published Figure 1c values, never on the same axis."""
    df, held = _STATE.get("data"), _STATE.get("held_out", "Ery")
    if df is None:
        raise SystemExit("Run step 1 first: the data are not loaded yet.")
    res = dict(_STATE.get("results", {}))
    train, test = _split(df, held)
    for key, feats in (("A", ["promoter"]), ("B", ["promoter", "nearby_50kb"])):
        if key not in res:                                          # fill anything the student skipped
            mdl = make_model().fit(train[feats], train.expression)
            res[key] = _metrics(test.expression, mdl.predict(test[feats]))
    res["control"] = _metrics(test.expression, test.average_in_training)
    # oracle: the single training cell type that best matches the held-out answer (chosen WITH the answer)
    wide = train.pivot(index="gene", columns="cell_type", values="expression").loc[test.gene]
    y = test.expression.to_numpy()
    scores = {c: np.corrcoef(wide[c].to_numpy(), y)[0, 1] for c in wide.columns}
    res["oracle_type"] = max(scores, key=scores.get)
    res["oracle"] = _metrics(y, wide[res["oracle_type"]].to_numpy())
    _STATE["results"] = res

    rows = [  # (shared concept, your value, colour, note above your bar)
        ("A: promoter openness", res["A"]["r"], PALE_RUST, ""),
        ("B: promoter + nearby openness", res["B"]["r"], RUST, ""),
        ("Control: copy each gene's\naverage from training", res["control"]["r"], GREY, ""),
        ("Oracle: best-matching training\ncell type (uses the answer)", res["oracle"]["r"], ORACLE,
         f"best match: {res['oracle_type']}"),
        ("GET foundation model", None, INK, ""),
    ]
    published = list(PUBLISHED_FIG1C.items())
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True, gridspec_kw={"wspace": 0.05})
    yy = np.arange(len(rows))[::-1]
    for ax, panel in ((axes[0], "you"), (axes[1], "paper")):
        for yv, (concept, mine, col, note), (paper_label, theirs) in zip(yy, rows, published):
            v = mine if panel == "you" else theirs
            above = note if panel == "you" else f"\u201c{paper_label}\u201d"
            if v is None:
                ax.text(0.02, yv, "not run in this lab", va="center", fontsize=10.5, color=MUTED,
                        style="italic")
                continue
            ax.barh(yv, v, color=col, height=0.56)
            ax.text(v + 0.012, yv, f"{v:.2f}", va="center", fontsize=12, fontweight="bold", color=INK)
            if above:
                ax.text(0.005, yv + 0.31, above, va="bottom", fontsize=9.5, color=MUTED, style="italic")
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Pearson r, held-out cell type")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_ylim(-0.5, len(rows) - 0.25)
    axes[0].set_title(f"You, today\nblood cells, {CELL_NAMES[held]} held out", loc="left", fontsize=12)
    axes[1].set_title("Fu et al. 2025, Figure 1c\nfetal astrocytes held out (published values)",
                      loc="left", fontsize=12)
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=10.5, color=INK)
    top = _title(fig, "Your experiment and Figure 1c ask the same question",
                 "The tissues, genes and models differ, so only the order of the bars is comparable. "
                 "Right: the values printed on the published panels, redrawn.", gap=0.75)
    fig.subplots_adjust(top=top, bottom=0.1, left=0.235, right=0.99)
    _show(fig)
    _html(_card("The same pattern",
                "In both experiments, accessibility alone loses to copying each gene's average expression. "
                "GET is the only real predictor that beats that control (0.94 against 0.78). The best-matching "
                f"training cell type beats it too (0.83 in the paper, {res['oracle']['r']:.2f} here), but only "
                "because it is picked by looking at the answer. GET also reads far more than our model: motif "
                "scores for 282 transcription-factor clusters in each of 200 neighbouring open regions, combined "
                "by a transformer pretrained on chromatin from 213 cell types.", "done"))
    _html(_card("How the two tests differ",
                "GET's astrocytes were held out of expression fine-tuning, but their chromatin was part of "
                f"pretraining. Our {CELL_NAMES[held]} were held out of all model training; their chromatin only "
                "helped define the shared list of open regions. The next part of the talk looks at how the "
                "authors evaluated GET.", "warn"))
    _html(f'<div style="color:{MUTED};background:{BG};padding:4px 10px;border-radius:6px;font-size:12px;'
          f'font-family:sans-serif;max-width:860px;">Published values: '
          f"{html.escape(CITATION)}, Figure 1c panel labels.</div>")


# --------------------------------------------------------------------------------------------
# optional, after class
# --------------------------------------------------------------------------------------------
def identity_genes(test, predicted, n=8):
    """Where chromatin beats the control: genes whose promoter opens most in the held-out cell type,
    picked from the inputs alone."""
    held, df = _STATE["held_out"], _STATE["data"]
    train_mean = df[df.cell_type != held].groupby("gene").promoter.mean()
    t = test.assign(model=np.asarray(predicted, float))
    t["gain"] = t.promoter - t.gene.map(train_mean)
    ranked = t.sort_values("gain", ascending=False)
    top = ranked.head(n)
    fig, ax = plt.subplots(figsize=(10, 0.55 * n + 1.9))
    yy = np.arange(len(top))[::-1]
    for yv, r in zip(yy, top.itertuples()):
        lo, hi = min(r.model, r.average_in_training, r.expression), max(r.model, r.average_in_training, r.expression)
        ax.plot([lo, hi], [yv, yv], color=GRID, lw=3, zorder=1)
    ax.scatter(top.expression, yy, s=90, color=INK, label="measured", zorder=3)
    ax.scatter(top.model, yy, s=90, color=RUST, label="accessibility model", zorder=3)
    ax.scatter(top.average_in_training, yy, s=90, color=GREY, label="copy-the-average control", zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels(top.gene, fontsize=11)
    ax.set_xlabel("Expression, log$_2$(RNA CPM + 1)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=10.5)
    ax.grid(axis="y", visible=False)
    ytop = _title(fig, f"Genes whose promoter opens most in {CELL_NAMES[held]}",
                  "Picked from the inputs alone: promoter openness here minus its average in the 15 training "
                  "cell types. The control cannot see that change; the accessibility model can.", gap=0.55)
    fig.subplots_adjust(top=ytop, left=0.12, right=0.98, bottom=0.13)
    _show(fig)

    def closer(frame):
        return int((abs(frame.model - frame.expression) < abs(frame.average_in_training - frame.expression)).sum())
    top50 = ranked.head(50)
    _note(f"The accessibility model is closer to the measurement than the control for {closer(top)} of these "
          f"{len(top)} genes, and for {closer(top50)} of the top 50.")


def biggest_misses(test, predicted, n=8):
    """The genes the model gets most wrong, in each direction."""
    t = test.assign(predicted=np.asarray(predicted, float))
    t["error"] = t.expression - t.predicted
    th = f'style="text-align:left;padding:5px 12px;border-bottom:2px solid {INK};color:{INK};"'
    td = f'style="padding:4px 12px;border-bottom:1px solid {GRID};color:{INK};text-align:left;"'
    tn = f'style="padding:4px 12px;border-bottom:1px solid {GRID};color:{INK};text-align:right;"'
    head = "".join(f"<th {th}>{h}</th>" for h in ["Gene", "Promoter openness", "Predicted", "Measured", "Off by"])

    def table(rows, title, colour):
        body = "".join(f"<tr><td {td}><b>{r.gene}</b></td><td {tn}>{r.promoter:.2f}</td>"
                       f"<td {tn}>{r.predicted:.2f}</td><td {tn}>{r.expression:.2f}</td>"
                       f"<td {tn}><b>{2 ** abs(r.error):,.0f}&times;</b></td></tr>" for r in rows.itertuples())
        return (f'<div style="display:inline-block;vertical-align:top;margin:0 18px 10px 0;">'
                f'<div style="font-weight:700;color:{colour};font-family:sans-serif;font-size:15px;'
                f'margin-bottom:4px;">{title}</div><table style="border-collapse:collapse;white-space:nowrap;'
                f'font-family:sans-serif;font-size:13.5px;background:{BG};">{head}{body}</table></div>')

    under, over = t.nlargest(n, "error"), t.nsmallest(n, "error")
    _html(_scroll(table(under, "Far more RNA than predicted", INK) + table(over, "Far less RNA than predicted", RUST)))
    lead = (" For erythroid precursors, HBB tops the list." if
            _STATE.get("held_out") == "Ery" and "HBB" in set(under.gene) else "")
    _html(_card("Reading the two lists",
                "Left: genes this cell type makes in enormous amounts. A straight line fitted to typical genes "
                "cannot reach such extreme values, so it under-predicts the genes that matter most to the "
                f"cell.{lead}<br>"
                "Right: promoters that are open but nearly silent. Open DNA means a gene <i>can</i> be read; "
                "some promoters stay open while the gene is switched off. A few of these genes read zero in all "
                "16 cell types, which can also come from how the original study counted reads (see "
                "docs/DATA_PROVENANCE.md). Values are log&#8322;(CPM + 1); &ldquo;off by&rdquo; is the fold "
                "difference."))


def try_cell_type(df, held_out):
    """Run the whole comparison for any held-out cell type."""
    train, test = _split(df, held_out)
    out = {}
    for key, feats in (("A: promoter only", ["promoter"]), ("B: + nearby (50 kb)", ["promoter", "nearby_50kb"])):
        mdl = make_model().fit(train[feats], train.expression)
        out[key] = _metrics(test.expression, mdl.predict(test[feats]))["r"]
    out["copy-the-average control"] = _metrics(test.expression, test.average_in_training)["r"]
    print(f"Held out: {CELL_NAMES[held_out]} ({len(test):,} genes)")
    for k, v in out.items():
        print(f"  {k:<26} r = {v:.3f}")


def all_cell_types(df):
    """Hold out each of the 16 cell types in turn and compare."""
    rows = []
    for ct in CELL_ORDER:
        train, test = _split(df, ct)
        row = {"cell type": ct}
        for key, feats in (("A", ["promoter"]), ("B", ["promoter", "nearby_50kb"])):
            mdl = make_model().fit(train[feats], train.expression)
            row[key] = _metrics(test.expression, mdl.predict(test[feats]))["r"]
        row["control"] = _metrics(test.expression, test.average_in_training)["r"]
        rows.append(row)
    res = pd.DataFrame(rows).set_index("cell type")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    x = np.arange(len(res))
    ax.scatter(x, res.control, s=70, color=GREY, label="copy-the-average control", zorder=3)
    ax.scatter(x, res.B, s=70, color=RUST, label="B: promoter + nearby", zorder=4)
    ax.scatter(x, res.A, s=28, color="white", edgecolor=RUST, linewidth=1.5, label="A: promoter only", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(res.index, rotation=45, ha="right")
    ax.set_ylabel("Pearson r, held-out cell type")
    ax.set_ylim(0.4, 1.0)
    ax.legend(loc="lower left", ncol=3, fontsize=10.5)
    d = res.B - res.A
    top = _title(fig, "Every cell type, held out in turn",
                 f"A: {res.A.min():.2f} to {res.A.max():.2f}. B minus A: {d.min():+.3f} to {d.max():+.3f}. "
                 f"Control: {res.control.min():.2f} to {res.control.max():.2f}.", gap=0.2)
    fig.subplots_adjust(top=top, bottom=0.2, left=0.08, right=0.99)
    _show(fig)
    return res.round(3)


def distance_sweep(df, held_out="Ery"):
    """How far is 'nearby'? Promoter plus openness within 10, 50, 100 or 250 kb."""
    train, test = _split(df, held_out)
    out = {}
    for feats in (["promoter"], ["promoter", "nearby_10kb"], ["promoter", "nearby_50kb"],
                  ["promoter", "nearby_100kb"], ["promoter", "nearby_250kb"],
                  ["promoter", "nearby_10kb", "nearby_50kb", "nearby_100kb", "nearby_250kb"]):
        mdl = make_model().fit(train[feats], train.expression)
        name = "promoter only" if len(feats) == 1 else (
            "promoter + all four windows" if len(feats) > 2 else f"promoter + {FEATURE_NAMES[feats[1]]}")
        out[name] = round(_metrics(test.expression, mdl.predict(test[feats]))["r"], 3)
    return pd.DataFrame({"Pearson r, held-out genes": out})


def flexible_model(train, test):
    """Replace the straight line with gradient-boosted decision trees. Does nearby openness help now?"""
    out = {}
    for name, feats in (("promoter only", ["promoter"]), ("promoter + nearby (50 kb)", ["promoter", "nearby_50kb"]),
                        ("promoter + all four windows", ["promoter", "nearby_10kb", "nearby_50kb",
                                                         "nearby_100kb", "nearby_250kb"])):
        lin = make_model().fit(train[feats], train.expression)
        trees = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, early_stopping=False,
                                              random_state=0).fit(train[feats], train.expression)
        out[name] = {"straight line": _metrics(test.expression, lin.predict(test[feats]))["r"],
                     "decision trees": _metrics(test.expression, trees.predict(test[feats]))["r"]}
    return pd.DataFrame(out).T.round(3)


def cell_type_specific(df, held_out="Ery"):
    """Separate 'which gene is this?' from 'what is special about this cell type?'."""
    train, test = _split(df, held_out)
    # each training row's usual level = its gene's average over the OTHER 14 training cell types,
    # so no row ever sees its own answer; test rows use the average over all 15 training cell types
    grp = train.groupby("gene").expression
    train = train.assign(usual=(grp.transform("sum") - train.expression) / (grp.transform("size") - 1),
                         promoter_change=train.promoter - train.groupby("gene").promoter.transform("mean"))
    prom_mean = train.groupby("gene").promoter.mean()
    test = test.assign(usual=test.average_in_training, promoter_change=test.promoter - test.gene.map(prom_mean))
    measured_change = (test.expression - test.usual).to_numpy()

    out = {}
    for name, feats in (("accessibility model (step 3)", ["promoter"]),
                        ("gene's usual level only", ["usual"]),
                        ("usual level + promoter + its change", ["usual", "promoter", "promoter_change"])):
        mdl = make_model().fit(train[feats], train.expression)
        pred = mdl.predict(test[feats])
        # the model's own usual prediction for each gene: its average over the 15 training cell types
        own = pd.Series(mdl.predict(train[feats]), index=train.index).groupby(train.gene).mean()
        change = pred - test.gene.map(own).to_numpy()
        specific = float(np.corrcoef(change, measured_change)[0, 1]) if np.std(change) > 1e-9 else 0.0
        out[name] = (_metrics(test.expression, pred)["r"], specific)
    res = pd.DataFrame(out, index=["overall r", "cell-type-specific r"]).T.round(3)
    _note("<b>Cell-type-specific r</b>: does the model's change for this cell type (its prediction minus its "
          "own average prediction for that gene) match the measured change (measured minus the gene's usual "
          "level)? A model whose prediction never changes between cell types scores 0.")
    return res


def all_genes_vs_open(df, held_out="Ery"):
    """Why GET (and we) score only genes with an open promoter."""
    train = df[df.cell_type != held_out]
    every = df[df.cell_type == held_out]
    mdl = make_model().fit(train[["promoter"]], train.expression)
    r_all = _metrics(every.expression, mdl.predict(every[["promoter"]]))["r"]
    opn = every[every.promoter >= OPEN]
    r_open = _metrics(opn.expression, mdl.predict(opn[["promoter"]]))["r"]
    closed = len(every) - len(opn)
    print(f"Scoring genes with an open promoter ({len(opn):,}):   r = {r_open:.3f}")
    print(f"Scoring every gene, closed ones too ({len(every):,}): r = {r_all:.3f}")
    _note(f"The {closed:,} closed-promoter genes are easy: closed DNA, almost no RNA. Including them "
          "makes the same model look better without it learning anything new.")


def pretraining_matters():
    """Published values only: GET's own ablation (Extended Data Fig. 2d)."""
    fig, ax = plt.subplots(figsize=(10, 4.0))
    names, vals = list(PUBLISHED_NO_PRETRAINING), list(PUBLISHED_NO_PRETRAINING.values())
    yy = np.arange(len(names))[::-1]
    ax.barh(yy, vals, color=[INK, GET_LIGHT, GET_LIGHT], height=0.6)
    for yv, v in zip(yy, vals):
        ax.text(v + 0.01, yv, f"{v:.2f}", va="center", fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 1.05)
    ax.axvline(0.78, color=GREY, ls="--", lw=1.4)
    ax.text(0.776, 2.45, "copy-the-average\ncontrol, 0.78", color=MUTED, fontsize=9.5, ha="right", va="bottom")
    ax.set_ylim(-0.5, 3.0)
    ax.set_xlabel("Pearson r, held-out fetal astrocytes")
    ax.grid(axis="y", visible=False)
    top = _title(fig, "Remove pretraining and GET falls below the copy-the-average control",
                 "Published values, Fu et al. 2025, Extended Data Fig. 2d (control line from the same panel), "
                 "redrawn.", gap=0.2)
    fig.subplots_adjust(top=top, bottom=0.17, left=0.33, right=0.97)
    _show(fig)
