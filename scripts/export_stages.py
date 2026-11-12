"""Export the five classroom results from the executed notebook.

    python scripts/export_stages.py [out_dir_for_slides]

Writes
  assets/stages/step1.png ... step5.png          the figures exactly as the notebook drew them
  assets/stages/step4_phone.png, step5_phone.png  narrow summaries of steps 4 and 5 for phone screens
  assets/stages/extra_*.png                       three figures from the optional section
  <out_dir>/fallback_step1.png ... step5.png      1920 x 1080 slides for the main deck, if Colab fails
Every number on the slides and phone images is read from the executed notebook's own output (or,
for the published values, from nebula_lab.py), never typed in here.
"""
import base64
import io
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nbformat  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nebula_lab as lab  # noqa: E402

BG, INK, TEAL, RUST, MUTED, GREY = lab.BG, lab.INK, lab.TEAL, lab.RUST, lab.MUTED, lab.GREY
W, H = 1920, 1080
FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")


def font(name, size):
    try:
        return ImageFont.truetype(str(FONT_DIR / name), size)
    except OSError:
        return ImageFont.load_default()


def outputs(cell):
    imgs, text = [], []
    for o in cell.outputs:
        if o.output_type == "display_data" and "image/png" in o.data:
            imgs.append(Image.open(io.BytesIO(base64.b64decode(o.data["image/png"]))).convert("RGB"))
        if o.output_type in ("display_data", "execute_result") and "text/html" in o.data:
            text.append(o.data["text/html"])
        if o.output_type == "stream":
            text.append(o.text)
    plain = re.sub(r"<[^>]+>", " ", " ".join(text))
    for a, b in (("&times;", "×"), ("&minus;", "−"), ("&sup2;", "²"), ("&Delta;", "Δ"),
                 ("&#8322;", "₂"), ("&rarr;", "→"), ("&nbsp;", " "), ("&beta;", "β")):
        plain = plain.replace(a, b)
    return imgs, re.sub(r"\s+", " ", plain)


def grab(pattern, text):
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"could not find {pattern!r} in the notebook output")
    return m.group(1)


def draw_metrics(d, items, x, y, width, vertical):
    """Big numbers with small labels. items: (value, label, colour)."""
    big, small = font("DejaVuSans-Bold.ttf", 66), font("DejaVuSans.ttf", 25)
    if vertical:
        for value, label, colour in items:
            d.text((x, y), value, font=big, fill=colour)
            for i, line in enumerate(textwrap.wrap(label, 30)):
                d.text((x, y + 80 + 31 * i), line, font=small, fill=MUTED)
            y += 80 + 31 * len(textwrap.wrap(label, 30)) + 34
        return y
    col = width // len(items)
    for k, (value, label, colour) in enumerate(items):
        d.text((x + k * col, y), value, font=big, fill=colour)
        for i, line in enumerate(textwrap.wrap(label, 34)):
            d.text((x + k * col, y + 80 + 31 * i), line, font=small, fill=MUTED)
    return y + 160


def slide(fig, step, items, message, footer, out_path, layout):
    canvas = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(canvas)
    d.text((70, 40), f"NEBULA LAB, step {step} of 5. A precomputed run of this notebook.",
           font=font("DejaVuSans-Bold.ttf", 23), fill=TEAL)
    if layout == "side":                        # figure left, numbers right
        box_w, box_h, top = 1180, 890, 95
        scale = min(box_w / fig.width, box_h / fig.height)
        f2 = fig.resize((int(fig.width * scale), int(fig.height * scale)), Image.LANCZOS)
        canvas.paste(f2, (60 + (box_w - f2.width) // 2, top + (box_h - f2.height) // 2))
        y = draw_metrics(d, items, 1300, 150, 560, vertical=True)
        for i, line in enumerate(textwrap.wrap(message, 34)):
            d.text((1300, y + 10 + 38 * i), line, font=font("DejaVuSans-Bold.ttf", 30), fill=INK)
    else:                                       # figure on top, numbers in a row underneath
        box_w, box_h, top = 1780, 700, 85
        scale = min(box_w / fig.width, box_h / fig.height)
        f2 = fig.resize((int(fig.width * scale), int(fig.height * scale)), Image.LANCZOS)
        canvas.paste(f2, ((W - f2.width) // 2, top + (box_h - f2.height) // 2))
        draw_metrics(d, items, 90, 800, 1740, vertical=False)
    d.text((70, H - 50), footer, font=font("DejaVuSans.ttf", 21), fill=MUTED)
    canvas.save(out_path, optimize=True)


def phone_bars(path, title, panels):
    """A narrow, large-type bar chart for phone screens. panels: [(heading, [(label, value, colour)])]."""
    n = sum(len(rows) for _, rows in panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(5.4, 1.15 + 0.8 * n + 0.75 * len(panels)),
                             gridspec_kw={"height_ratios": [len(r) for _, r in panels], "hspace": 0.42},
                             squeeze=False)
    for ax, (heading, rows) in zip(axes[:, 0], panels):
        yy = list(range(len(rows)))[::-1]
        for yv, (label, value, colour) in zip(yy, rows):
            shown = value if isinstance(value, str) else f"{value:.2f}"
            ax.barh(yv, float(value), color=colour, height=0.52)
            ax.text(float(value) + 0.02, yv, shown, va="center", fontsize=13, fontweight="bold", color=INK)
            ax.text(0.01, yv + 0.33, label, va="bottom", fontsize=10.5, color=INK)
        ax.set_xlim(0, 1.12)
        ax.set_ylim(-0.5, len(rows) - 0.05)
        ax.set_yticks([])
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_title(heading, loc="left", fontsize=11.5, color=INK)
        ax.grid(axis="y", visible=False)
    axes[-1, 0].set_xlabel("Pearson r, held-out cell type")
    fig.suptitle(title, x=0.02, ha="left", fontsize=13.5, fontweight="bold", family="DejaVu Serif", color=INK)
    fig.subplots_adjust(left=0.04, right=0.97, top=1 - 0.75 / fig.get_size_inches()[1], bottom=0.12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(slide_dir):
    nb = nbformat.read(str(ROOT / "NEBULA_Lab_completed.ipynb"), as_version=4)
    code = [c for c in nb.cells if c.cell_type == "code"]
    got = [outputs(c) for c in code[:5]]
    stages = ROOT / "assets" / "stages"
    stages.mkdir(parents=True, exist_ok=True)
    slide_dir.mkdir(parents=True, exist_ok=True)
    t1, t2, t3, t4, t5 = (g[1] for g in got)

    genes, rows = grab(r"([0-9,]+) genes × 16", t1), grab(r"= ([0-9,]+) measurements", t1)
    n_train = grab(r"learned from ([0-9,]+) examples", t2)
    equation = grab(r"predicted expression = (.+?)\s*$", t2)
    r3, r2_3 = grab(r"r = ([0-9.]+)", t3), grab(r"R² = ([0-9.-]+)", t3)
    med3, below3 = grab(r"median error is ([0-9.]+)-fold", t3), grab(r"([0-9]+)% of genes sit below", t3)
    ra, rb = re.search(r"r went from ([0-9.]+) to ([0-9.]+)", t4).groups()
    rc = grab(r"reaches r = ([0-9.]+)", t4)
    oracle = grab(r"0\.83 in the paper, ([0-9.]+) here", t5)
    pub = lab.PUBLISHED_FIG1C
    main_footer = ("Real ATAC-seq and RNA-seq of 16 sorted human blood populations (Corces et al. 2016; "
                   "Ulirsch et al. 2019). A simplified educational model, not GET.")

    specs = {
        1: ("stack", [(genes, "genes", INK), ("16", "blood cell types", TEAL), (rows, "real measurements", INK)],
            "", main_footer),
        2: ("side", [(n_train, "training examples from 15 cell types", INK),
                     ("1 line", f"predicted expression = {equation}", RUST)],
            "The model never saw a single erythroid row.", main_footer),
        3: ("side", [(f"r = {r3}", "predictions rise with the measurements", RUST),
                     (f"R² = {r2_3}", "how close they are (1 is perfect)", INK),
                     (f"{below3}%", "of genes over-predicted: too high on average", GREY)],
            f"Half the genes are within {med3}-fold.", main_footer),
        4: ("side", [(f"A {ra}", "promoter features only", RUST), (f"B {rb}", "promoter + nearby (50 kb)", RUST),
                     (f"{rc}", "control: copy each gene's average", GREY)],
            "Nearby regions: no meaningful difference. The control beats both.", main_footer),
        5: ("stack", [(f"{float(ra):.2f}", "you: accessibility alone", RUST), (rc, "you: copy the average", GREY),
                      (f"{pub['Mean expression in training set']:.2f}", "paper: copy the average (published)", GREY),
                      (f"{pub['GET prediction']:.2f}", "paper: GET (published)", INK)],
            "", "Left of the figure: computed in this lab. Right: published values, Fu et al., Nature 2025, "
                "Fig. 1c, redrawn."),
    }
    for i, (imgs, _) in enumerate(got, start=1):
        if not imgs:
            raise SystemExit(f"step {i} produced no figure")
        imgs[0].save(stages / f"step{i}.png", optimize=True)
        layout, items, message, footer = specs[i]
        slide(imgs[0], i, items, message, footer, slide_dir / f"fallback_step{i}.png", layout)
        print(f"step {i}: " + "; ".join(f"{v} {lbl}" for v, lbl, _ in items))

    phone_bars(stages / "step4_phone.png", "Step 4: does nearby DNA help?",
               [("Erythroid precursors, held out", [("A: promoter features only", ra, lab.PALE_RUST),
                                                    ("B: promoter + nearby (50 kb)", rb, RUST),
                                                    ("Control: copy each gene's average", rc, GREY)])])
    phone_bars(stages / "step5_phone.png", "Step 5: you and Figure 1c",
               [("You: blood cells, erythroid precursors held out",
                 [("A: promoter openness", float(ra), lab.PALE_RUST), ("B: + nearby openness", float(rb), RUST),
                  ("Control: copy the average", float(rc), GREY),
                  ("Oracle: best training type (uses the answer)", float(oracle), lab.ORACLE)]),
                ("Fu et al. 2025, Fig. 1c, published: fetal astrocytes",
                 [("TSS accessibility", pub["TSS accessibility"], lab.PALE_RUST),
                  ("Gene activity", pub["Gene activity"], RUST),
                  ("Mean expression in training set", pub["Mean expression in training set"], GREY),
                  ("Top correlated cell type (uses the answer)", pub["Top correlated cell type"], lab.ORACLE),
                  ("GET prediction", pub["GET prediction"], INK)])])
    print("phone images: step4_phone.png, step5_phone.png")

    extras = {"lab.identity_genes(": "extra_identity_genes.png", "lab.all_cell_types(": "extra_all_cell_types.png",
              "lab.pretraining_matters(": "extra_pretraining.png"}
    for c in nb.cells:
        for call, name in extras.items():
            if c.cell_type == "code" and call in c.source:
                imgs, _ = outputs(c)
                imgs[0].save(stages / name, optimize=True)
                print("optional figure:", name)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "backup" / "fallback_slides")
