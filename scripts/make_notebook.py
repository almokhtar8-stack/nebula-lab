"""Write NEBULA_Lab.ipynb, the classroom notebook.

    python3 scripts/make_notebook.py            # writes NEBULA_Lab.ipynb next to nebula_lab.py

The completed version (all outputs saved) is produced by executing this notebook:
see scripts/run_and_test.py.
"""
import html
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent.parent
REPO = "almokhtar8-stack/nebula-lab"
REF = "v1.0"                      # a fixed release, so every student downloads identical files
RAW = f"https://raw.githubusercontent.com/{REPO}/{REF}/"
CDN = f"https://cdn.jsdelivr.net/gh/{REPO}@{REF}/"
REPO_URL = f"https://github.com/{REPO}"
COMPLETED = f"https://github.com/{REPO}/blob/{REF}/NEBULA_Lab_completed.ipynb"

INK, TEAL, RUST, BG = "#123E38", "#17645B", "#C45B39", "#FAFCF8"
SANS = "Arial,Helvetica,sans-serif"
SERIF = "Georgia,'Times New Roman',serif"


def md(text):
    return new_markdown_cell(text.strip("\n"))


def code(text, form=False):
    cell = new_code_cell(text.strip("\n"))
    if form:
        cell.metadata["cellView"] = "form"
    return cell


WELCOME = f"""
<div style="background:{INK};border-radius:4px;padding:24px 28px 20px;color:{BG};">
<div style="font-family:{SANS};font-size:13px;line-height:1.4;color:#A9CFC5;">KAUST B224 journal club &middot; Project Nebula</div>
<div style="font-family:{SERIF};font-size:44px;line-height:1.1;font-weight:700;margin:8px 0 6px;color:{BG};">NEBULA LAB</div>
<div style="font-family:{SERIF};font-size:21px;line-height:1.3;color:#E3EEEA;">Build Your Own Gene Expression Predictor</div>
<div style="font-family:{SANS};font-size:14px;line-height:1.4;color:#CFE3DD;margin-top:14px;">&#9312; Data &rarr; &#9313; Train &rarr; &#9314; Predict &rarr; &#9315; Experiment &rarr; &#9316; Discover</div>
</div>

<p style="font-size:17px;margin:16px 0 6px;"><b>Can a computer predict how strongly genes are expressed using regulatory information? Let's find out.</b></p>

This is a simplified educational model. It is not the GET foundation model. The measurements are real; the model is one straight line.

Run each numbered cell from top to bottom: click &#9654; at the cell's left, or press Shift + Enter. The first run can take 10 to 20 seconds while Colab starts a computer for you. If Colab warns that the notebook was not authored by Google, click Run anyway.

<sub>On a phone or without a Google account, see every result on [the project page]({REPO_URL}) or in the [completed notebook]({COMPLETED}).</sub>
"""

STEP1_MD = """
### &#9312; Explore the biological data
Real measurements from 16 kinds of human blood cells, each sorted and sequenced in bulk. ATAC-seq measures how *open* the DNA is around each gene; RNA-seq measures how much *RNA* each gene makes. Can the first predict the second?
"""

STEP1 = f"""
#@title ① Load the data {{ display-mode: "form" }}
# Fetches the lab's helper code and its 3.9 MB data table from GitHub.
import os, urllib.request
for base in ("{RAW}", "{CDN}"):
    if os.path.exists("nebula_lab.py"):
        break
    try:
        code = urllib.request.urlopen(base + "nebula_lab.py", timeout=30).read()
        if b"def meet_get" in code:                    # a complete copy, not an error page
            open("nebula_lab.py", "wb").write(code)
    except Exception:
        pass
if not os.path.exists("nebula_lab.py"):
    raise SystemExit("Could not download the lab files. Follow along on the projector, "
                     "or see every result on the project page: {REPO_URL}")
import nebula_lab as lab
data = lab.load_data()
lab.show_data(data)
"""

STEP2_MD = """
### &#9313; Train the model
We hide one cell type, erythroid precursors (the cells that become red blood cells), and train on the other 15. For every gene in those 15 cell types the model sees promoter openness and measured expression, and fits the straight line that links them best.
"""

STEP2 = """
#@title ② Train the model
HELD_OUT = "Ery"                                       # erythroid precursors: kept aside, never seen
train, test = lab.split(data, HELD_OUT)                # 15 cell types to learn from, 1 to test on

model = lab.make_model()                               # a straight-line model (ridge regression)
model.fit(train[["promoter"]], train["expression"])    # learning: promoter openness -> expression

lab.show_training(model, train)
"""

STEP3_MD = """
### &#9314; Can the model predict expression?
The model now predicts expression in erythroid precursors, a cell type it never saw, for every gene with an open promoter there. Each prediction is compared with what was measured.

Before you run it: will the dots fall along the diagonal?
"""

STEP3 = """
#@title ③ Predict and test
predicted = model.predict(test[["promoter"]])          # predictions for the cell type it never saw
lab.show_predictions(test, predicted)
"""

STEP4_MD = """
### &#9315; Let's try to improve it
Genes are also controlled by enhancers, which can sit tens of kilobases from the promoter.

* Option A: promoter features only. The step 3 model.
* Option B: promoter + nearby regulatory features. Promoter openness plus the openness of all other open regions within 50 kb of the gene's start sites.

Will B beat A? Make a guess, then run the cell. The menu is already set to option B.
"""

STEP4 = """
#@title ④ Experiment: does nearby regulatory DNA help?
choice = "Option B: promoter + nearby regulatory features"  #@param ["Option A: promoter features only", "Option B: promoter + nearby regulatory features"]

features = lab.FEATURE_SETS[choice]                     # the inputs you chose
new_model = lab.make_model().fit(train[features], train["expression"])
lab.compare(test, predicted, new_model.predict(test[features]), features, new_model)
"""

STEP5_MD = f"""
### &#9316; Now meet GET
GET, the *general expression transformer* (Fu *et al.*, *Nature* 2025), asks the same question at a much larger scale.

| | Your model | GET |
|:---|:---|:---|
| What it reads | 1 or 2 numbers per gene: how open the DNA is | a window of 200 neighbouring open regions, each described by scores for 282 clusters of transcription-factor motifs plus its openness |
| The model | a straight line | a 12-layer transformer whose *attention* weighs every region against every other |
| How it learns | from 274,275 examples in 15 blood cell types | first *pretrained* on chromatin from 213 cell types by hiding parts of the input and filling them back in, then *fine-tuned* to predict expression, using the cell types that also had RNA data (153 in all) |
| The test | erythroid precursors, held out of all training | fetal astrocytes, held out of the expression fine-tuning stage |

This is the published Figure 1c. Each dot is a gene in fetal astrocytes, laid out like your plot: prediction along the bottom, measured expression up the side. GET is top left; the other four panels are the comparisons the authors chose. The bottom row uses accessibility alone, as your model does.

<img src="{RAW}assets/fu2025_fig1c.png" width="760" alt="Fu et al. 2025 Figure 1c">

<sub>Fu X, Mo S, Buendia A, <i>et al.</i> A foundation model of transcription across human cell types. <i>Nature</i> 637:965&ndash;973 (2025), Figure 1c, doi:<a href="https://doi.org/10.1038/s41586-024-08391-z">10.1038/s41586-024-08391-z</a>. Shown unaltered (cropped from Figure 1) under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">CC BY-NC-ND 4.0</a>. This is the published figure, not a reproduction. The R&sup2; printed on each panel is the square of r. The R&sup2; in your notebook measures how close the predictions are to the measurements, so the two cannot be compared.</sub>

Run the next cell to put your results beside theirs.
"""

STEP5 = """
#@title ⑤ Compare your experiment with GET { display-mode: "form" }
lab.meet_get()
"""

TRANSITION = """
**What you just built is a simple predictor. Now let's see how the researchers approached a much more difficult version of this problem.**
"""

OPTIONAL_HEAD = """
## Optional: explore after class
Ten short experiments on the same data, for after the talk. Run steps &#9312; to &#9315; first.
"""

OPTIONAL = [
    ("1 &middot; Where does chromatin beat the control?",
     "The copy-the-average control wins overall because most genes behave alike in every blood cell type. "
     "It cannot see a promoter that opens only in the held-out cell type. These are the genes whose promoter "
     "opens most there compared with the training cell types, picked from the inputs alone.",
     "lab.identity_genes(test, predicted)"),
    ("2 &middot; Try another held-out cell type",
     "Pick any of the 16 cell types to hide instead. Is erythroid typical?",
     'other = "B"  #@param ["HSC", "MPP", "LMPP", "CMP", "GMP-A", "GMP-B", "GMP-C", "Mono", "pDC", "MEP", "Ery", "CLP", "B", "CD4", "CD8", "NK"]\n'
     "lab.try_cell_type(data, other)"),
    ("3 &middot; Every cell type at once",
     "Hold out each of the 16 cell types in turn and train both models every time (about 2 seconds).",
     "lab.all_cell_types(data)"),
    ("4 &middot; How far is &ldquo;nearby&rdquo;?",
     "The table also holds openness within 10, 100 and 250 kb. Does looking further help?",
     "lab.distance_sweep(data, HELD_OUT)"),
    ("5 &middot; A more flexible model",
     "A straight line cannot bend; gradient-boosted decision trees can. Does that flexibility let nearby "
     "regions help? (about 5 seconds)",
     "lab.flexible_model(train, test)"),
    ("6 &middot; Which gene, or which cell type?",
     "A prediction can score well because it knows *which gene* this is (housekeeping genes are high "
     "everywhere) or because it knows *what is special about this cell type*. The cell-type-specific score "
     "checks only the second: does the model's change for this cell type match the measured change? Here a "
     "model may also see each gene's usual expression level, so it no longer predicts from chromatin alone.",
     "lab.cell_type_specific(data, HELD_OUT)"),
    ("7 &middot; Calculate Pearson's r yourself",
     "The formula written out, checked against NumPy.",
     "import numpy as np\n"
     "y = test[\"expression\"].to_numpy()          # measured\n"
     "p = predicted                               # predicted\n"
     "r = ((y - y.mean()) * (p - p.mean())).sum() / np.sqrt(((y - y.mean())**2).sum() * ((p - p.mean())**2).sum())\n"
     "print(f\"by hand: r = {r:.4f}    NumPy: r = {np.corrcoef(y, p)[0, 1]:.4f}\")"),
    ("8 &middot; The genes your model gets most wrong",
     "A positive error means the gene makes far more RNA than its promoter openness suggests.",
     "lab.biggest_misses(test, predicted)"),
    ("9 &middot; Why score only open promoters?",
     "GET scores genes with an accessible promoter, and so does this lab. What happens if every gene counts?",
     "lab.all_genes_vs_open(data, HELD_OUT)"),
    ("10 &middot; Why pretraining matters",
     "The authors also trained the GET architecture without its pretraining stage. These are their published "
     "values; this notebook does not compute them.",
     "lab.pretraining_matters()"),
]

CREDITS = """
#### Where everything comes from
The data are bulk ATAC-seq and RNA-seq of 16 sorted human blood cell populations (Corces *et al.*, *Nature Genetics* 48:1193&ndash;1203, 2016; Buenrostro *et al.*, *Cell* 173:1535&ndash;1548, 2018), as processed in the public analysis repository of Ulirsch *et al.*, *Nature Genetics* 51:683&ndash;693 (2019) (github.com/caleblareau/singlecell_bloodtraits, commit `1da2a246`), and summarised per gene by this project. `docs/DATA_PROVENANCE.md` in the repository lists every step.

Values are log&#8322;(1 + CPM), where CPM is counts per million reads (for ATAC, reads in open regions). The promoter is the set of open regions within 1 kb of a start site of the gene; nearby means all other open regions within 10, 50, 100 or 250 kb of its start sites.

Figure 1c and every published value come from Fu *et al.*, *Nature* 637:965&ndash;973 (2025), CC BY-NC-ND 4.0. They are quoted and labelled as published, never recomputed. Everything else in this notebook is computed when you run it.
"""


def build(path):
    cells = [md(WELCOME),
             md(STEP1_MD), code(STEP1, form=True),
             md(STEP2_MD), code(STEP2),
             md(STEP3_MD), code(STEP3),
             md(STEP4_MD), code(STEP4),
             md(STEP5_MD), code(STEP5, form=True),
             md(TRANSITION),
             md(OPTIONAL_HEAD)]
    for title, text, body in OPTIONAL:
        cells.append(md(f"#### {title}\n{text}"))
        cells.append(code(f"#@title {html.unescape(title.split(' &middot; ')[1])}\n{body}"))
    cells.append(md(CREDITS))
    for i, cell in enumerate(cells):          # fixed ids, so a rebuild is byte-identical
        cell.id = f"nebula-{i:02d}"
    nb = new_notebook(cells=cells)
    nb.metadata = {
        "colab": {"name": "NEBULA_Lab.ipynb", "provenance": [], "toc_visible": False},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
    }
    nbformat.validate(nb)
    nbformat.write(nb, str(path))
    return nb


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "NEBULA_Lab.ipynb"
    nb = build(out)
    n_code = sum(c.cell_type == "code" for c in nb.cells)
    print(f"wrote {out}: {len(nb.cells)} cells ({n_code} code)")
