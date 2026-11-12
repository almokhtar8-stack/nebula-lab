"""Execute the classroom notebook end to end, check every result, and save the completed copy.

    python scripts/run_and_test.py [--latest-env-label NAME]

Runs in fresh temporary folders (like a new Colab session) and writes
  NEBULA_Lab.ipynb              the classroom notebook (no outputs), rebuilt from make_notebook.py
  NEBULA_Lab_completed.ipynb    every cell executed, outputs saved (phone and fallback copy)
  test/results_<env>.json       machine-readable results of every check
The checks recompute the key numbers independently, straight from the CSV with NumPy and
scikit-learn, and compare them with what the notebook printed.
"""
import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
from make_notebook import build  # noqa: E402

OPTION_A = "Option A: promoter features only"
OPTION_B = "Option B: promoter + nearby regulatory features"
CORE_CELLS = 11          # welcome ... step 5 code cell (indices 0-10)
RESULTS = {"checks": [], "timings": {}}


def check(name, ok, detail=""):
    RESULTS["checks"].append({"check": name, "pass": bool(ok), "detail": str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


def fresh_dir(with_module=True, with_data=True):
    d = Path(tempfile.mkdtemp(prefix="nebula_run_"))
    if with_module:
        shutil.copy(ROOT / "nebula_lab.py", d / "nebula_lab.py")
    if with_data:
        (d / "data").mkdir()
        shutil.copy(ROOT / "data" / "nebula_blood_atac_rna.csv.gz", d / "data")
    return d


def execute(nb, workdir, allow_errors=False):
    client = NotebookClient(nb, timeout=900, kernel_name="python3", allow_errors=allow_errors,
                            record_timing=True, resources={"metadata": {"path": str(workdir)}})
    t0 = time.time()
    client.execute()
    return time.time() - t0


def cell_seconds(cell):
    ex = cell.get("metadata", {}).get("execution", {})
    a, b = ex.get("iopub.execute_input"), ex.get("shell.execute_reply")
    if not (a and b):
        return None
    f = "%Y-%m-%dT%H:%M:%S.%fZ"
    return (datetime.strptime(b, f) - datetime.strptime(a, f)).total_seconds()


def outputs_text(cell):
    """All text and HTML a cell printed, tags stripped."""
    out = []
    for o in cell.get("outputs", []):
        if o.output_type == "stream":
            out.append(o.text)
        elif o.output_type in ("display_data", "execute_result"):
            out.append(o.data.get("text/html", o.data.get("text/plain", "")))
        elif o.output_type == "error":
            out.append(f"ERROR {o.ename}: {o.evalue}")
    return re.sub(r"<[^>]+>", " ", "\n".join(out))


def fingerprint(nb):
    """Hash of every output (text, HTML and image bytes), for determinism checks."""
    h = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        parts = []
        for o in c.outputs:
            if o.output_type == "stream":
                parts.append(o.text)
            elif o.output_type in ("display_data", "execute_result"):
                for k in sorted(o.data):
                    parts.append(k + str(o.data[k]))
        h.append(hashlib.sha256("".join(parts).encode()).hexdigest()[:16])
    return h


def set_choice(nb, choice):
    for c in nb.cells:
        if c.cell_type == "code" and c.source.lstrip().startswith("#@title \u2463"):
            c.source = re.sub(r'choice = "[^"]+"', f'choice = "{choice}"', c.source)
            return
    raise AssertionError("step 4 cell not found")


def independent_numbers():
    """Recompute the headline results without nebula_lab.py."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(ROOT / "data" / "nebula_blood_atac_rna.csv.gz")
    tr = df[df.cell_type != "Ery"]
    te = df[(df.cell_type == "Ery") & (df.promoter >= 1.0)]
    out = {"n_test": len(te), "n_train": len(tr)}
    for key, feats in (("A", ["promoter"]), ("B", ["promoter", "nearby_50kb"])):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(tr[feats], tr.expression)
        p = m.predict(te[feats])
        y = te.expression.to_numpy()
        out[f"r_{key}"] = float(np.corrcoef(y, p)[0, 1])
        out[f"R2_{key}"] = float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        out[f"fold_{key}"] = float(2 ** np.median(np.abs(y - p)))
        out[f"rmse_{key}"] = float(np.sqrt(((y - p) ** 2).mean()))
        out[f"bias_{key}"] = float((y - p).mean())
        out[f"below_{key}"] = float((y < p).mean())
    avg = tr.groupby("gene").expression.mean()
    out["r_control"] = float(np.corrcoef(te.expression, te.gene.map(avg))[0, 1])
    wide = tr.pivot(index="gene", columns="cell_type", values="expression").loc[te.gene]
    best = {c: float(np.corrcoef(wide[c], te.expression)[0, 1]) for c in wide.columns}
    out["oracle_type"] = max(best, key=best.get)
    out["r_oracle"] = best[out["oracle_type"]]
    return out


def static_checks(nb_path):
    print("\nStatic checks on the classroom notebook")
    nb = nbformat.read(str(nb_path), as_version=4)
    nbformat.validate(nb)
    check("notebook is valid nbformat 4", True)
    size = nb_path.stat().st_size
    check("classroom notebook is small (no data inside)", size < 60_000, f"{size:,} bytes")
    check("classroom notebook has no saved outputs",
          all(not c.get("outputs") for c in nb.cells if c.cell_type == "code"))
    longest = max((len(t) for c in nb.cells for t in re.findall(r"\S+", c.source)), default=0)
    check("no encoded data blocks in any cell", longest < 400, f"longest unbroken token: {longest} chars")
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    core = [c for c in nb.cells[:CORE_CELLS] if c.cell_type == "code"]
    check("five numbered classroom code cells", len(core) == 5, f"{len(core)} core, {len(code_cells)} total")
    longest_core = max(len(c.source.splitlines()) for c in core[1:4])
    check("visible classroom cells are short", longest_core <= 8, f"longest: {longest_core} lines")
    text = "\n".join(c.source for c in nb.cells) + (ROOT / "nebula_lab.py").read_text()
    check("no em dashes in student-facing text", "\u2014" not in text)
    banned = [w for w in ("leverage", "robust", "seamless", "elevate") if re.search(rf"\b{w}", text, re.I)]
    check("house-style word list respected", not banned, ", ".join(banned))
    import nebula_lab as lab
    opts = re.findall(r'#@param \[(.+)\]', nb.cells[8].source)[0]
    check("dropdown options match the lab's feature sets",
          sorted(re.findall(r'"([^"]+)"', opts)) == sorted(lab.FEATURE_SETS), opts)
    for c in code_cells:
        m = re.search(r'#@param \[(.+)\]', c.source)
        if m and "HSC" in m.group(1):
            check("cell-type dropdown lists all 16 cell types",
                  sorted(re.findall(r'"([^"]+)"', m.group(1))) == sorted(lab.CELL_ORDER))


def main(env_label):
    nb_path = ROOT / "NEBULA_Lab.ipynb"
    build(nb_path)
    static_checks(nb_path)
    ref = independent_numbers()
    RESULTS["independent"] = ref
    RESULTS["environment"] = {"label": env_label, "python": platform.python_version()}
    import matplotlib, numpy, pandas, sklearn
    RESULTS["environment"].update(numpy=numpy.__version__, pandas=pandas.__version__,
                                  sklearn=sklearn.__version__, matplotlib=matplotlib.__version__)
    print(f"\nEnvironment: {RESULTS['environment']}")

    # 1. the whole notebook, default choice (option B), in a fresh folder
    print("\nRun 1: whole notebook, default menu choice (option B), fresh folder")
    nb1 = nbformat.read(str(nb_path), as_version=4)
    wall1 = execute(nb1, fresh_dir())
    RESULTS["timings"]["whole_notebook_s"] = round(wall1, 1)
    per = [cell_seconds(c) for c in nb1.cells if c.cell_type == "code"]
    core_s = sum(s for s in per[:5] if s)
    RESULTS["timings"]["core_steps_s"] = [round(s, 2) for s in per[:5]]
    RESULTS["timings"]["optional_s"] = [round(s, 2) for s in per[5:]]
    check("every cell ran without an error", not any(o.output_type == "error" for c in nb1.cells
                                                     if c.cell_type == "code" for o in c.outputs))
    check("no warnings or stderr output",
          not any(o.output_type == "stream" and o.name == "stderr" for c in nb1.cells
                  if c.cell_type == "code" for o in c.outputs))
    check("five classroom steps run in under 20 s of compute", core_s < 20,
          f"{core_s:.1f} s (steps: {', '.join(f'{s:.1f}' for s in per[:5])})")

    code = [c for c in nb1.cells if c.cell_type == "code"]
    s1, s2, s3, s4, s5 = (outputs_text(c) for c in code[:5])
    check("step 1 reports the full table", "18,285 genes" in s1 and "292,560 measurements" in s1)
    check("step 1 draws one figure", sum(o.output_type == "display_data" and "image/png" in o.data
                                         for o in code[0].outputs) == 1)
    check("step 2 trained on every training row", f"{ref['n_train']:,} examples" in s2, f"{ref['n_train']:,}")
    m = re.search(r"r = ([0-9.]+)", s3)
    check("step 3 r matches independent recomputation", m and abs(float(m.group(1)) - ref["r_A"]) < 0.005,
          f"printed {m.group(1) if m else None}, independent {ref['r_A']:.4f}")
    m = re.search(r"R&sup2; = (-?[0-9.]+)|R² = (-?[0-9.]+)", s3)
    printed_r2 = float(next(g for g in m.groups() if g)) if m else None
    check("step 3 R^2 matches", printed_r2 is not None and abs(printed_r2 - ref["R2_A"]) < 0.005,
          f"printed {printed_r2}, independent {ref['R2_A']:.4f}")
    m = re.search(r"median error is ([0-9.]+)-fold", s3)
    check("step 3 median error matches", m and abs(float(m.group(1)) - ref["fold_A"]) < 0.05,
          f"printed {m.group(1) if m else None}, independent {ref['fold_A']:.3f}")
    m = re.search(r"([0-9]+)% of genes sit below the dashed line", s3)
    check("step 3 calibration statement matches", m and abs(int(m.group(1)) - 100 * ref["below_A"]) < 0.6,
          f"printed {m.group(1) if m else None}%, independent {100 * ref['below_A']:.1f}%")
    m = re.search(r"r went from ([0-9.]+) to ([0-9.]+)", s4)
    check("step 4 compares A and B with the right numbers",
          m and abs(float(m.group(1)) - ref["r_A"]) < 5e-4 and abs(float(m.group(2)) - ref["r_B"]) < 5e-4,
          f"printed {m.groups() if m else None}, independent {ref['r_A']:.3f} / {ref['r_B']:.3f}")
    d = ref["r_B"] - ref["r_A"]
    verdict = "no meaningful difference" if abs(d) < 0.01 else ("improved" if d > 0 else "worse")
    check("step 4 verdict follows the data, not a script", verdict in s4, verdict)
    m = re.search(r"reaches\s+r = ([0-9.]+)", s4)
    check("step 4 control value matches", m and abs(float(m.group(1)) - ref["r_control"]) < 0.005,
          f"printed {m.group(1) if m else None}, independent {ref['r_control']:.4f}")
    check("step 5 quotes GET's published r = 0.94 and labels it published",
          "0.94" in s5 and "published" in s5.lower())

    # completed copy: outputs from run 1
    done = (ROOT / "NEBULA_Lab_completed.ipynb" if env_label == "colab-matched"
            else ROOT / "test" / f"completed_{env_label}.ipynb")
    done.parent.mkdir(exist_ok=True)
    nb_done = nbformat.from_dict(json.loads(json.dumps(nb1)))
    for c in nb_done.cells:
        c.metadata.pop("execution", None)
    nb_done.metadata["colab"]["name"] = "NEBULA_Lab_completed.ipynb"
    welcome = nb_done.cells[0].source.split("\n")
    welcome = [ln for ln in welcome if not ln.startswith("<sub>On a phone")]
    i = next(k for k, ln in enumerate(welcome) if ln.startswith("Run each numbered cell"))
    welcome[i] = ("This copy has already been run: every result below is saved, so nothing needs to run here. "
                  "To run it yourself, open `NEBULA_Lab.ipynb` in Colab.")
    nb_done.cells[0].source = "\n".join(welcome).rstrip()
    nbformat.write(nb_done, str(done))
    RESULTS["completed_notebook_bytes"] = done.stat().st_size
    check("completed copy saved", done.exists(), f"{done.stat().st_size / 1e6:.1f} MB")

    # 2. determinism: the same notebook again, new folder, new kernel
    print("\nRun 2: identical rerun in a new folder and kernel")
    nb2 = nbformat.read(str(nb_path), as_version=4)
    execute(nb2, fresh_dir())
    f1, f2 = fingerprint(nb1), fingerprint(nb2)
    same = sum(a == b for a, b in zip(f1, f2))
    check("rerun reproduces every output exactly (text, tables and images)", f1 == f2, f"{same}/{len(f1)} cells identical")

    # 3. option A in the menu
    print("\nRun 3: menu set to option A")
    nb3 = nbformat.read(str(nb_path), as_version=4)
    set_choice(nb3, OPTION_A)
    nb3.cells = nb3.cells[:CORE_CELLS]
    execute(nb3, fresh_dir())
    s4a = outputs_text([c for c in nb3.cells if c.cell_type == "code"][3])
    check("option A reproduces step 3 exactly and says so", "Same model, same answer" in s4a)
    check("option A run has no errors", not any(o.output_type == "error" for c in nb3.cells
                                                if c.cell_type == "code" for o in c.outputs))

    # 4. step 5 straight after step 3 (a student who skipped the experiment)
    print("\nRun 4: skip step 4, run step 5")
    nb4 = nbformat.read(str(nb_path), as_version=4)
    del nb4.cells[7:9]
    nb4.cells = nb4.cells[:CORE_CELLS - 2]
    execute(nb4, fresh_dir())
    s5b = outputs_text(nb4.cells[-1])
    check("step 5 fills in the skipped experiment itself", "ERROR" not in s5b and "0.94" in s5b)

    # 5. no data on disk and no network: friendly message, no traceback wall
    print("\nRun 5: data missing and download blocked")
    nb5 = nbformat.read(str(nb_path), as_version=4)
    nb5.cells = nb5.cells[:3]
    wd = fresh_dir(with_data=False)
    mod = (wd / "nebula_lab.py").read_text().replace('"https://raw.githubusercontent.com/{repo}/{ref}/{path}"',
                                                     '"http://127.0.0.1:9/{path}"').replace(
        '"https://cdn.jsdelivr.net/gh/{repo}@{ref}/{path}"', '"http://127.0.0.1:9/b/{path}"')
    (wd / "nebula_lab.py").write_text(mod)
    execute(nb5, wd, allow_errors=True)
    s = outputs_text(nb5.cells[2])
    check("missing data gives the friendly 'could not be downloaded' card",
          "could not be downloaded" in s and "project page" in s)
    err = [o for o in nb5.cells[2].outputs if o.output_type == "error"]
    check("missing data stops quietly, without a long traceback",
          bool(err) and err[0].ename == "SystemExit" and len(err[0].traceback) <= 3,
          f"{err[0].ename}, {len(err[0].traceback)} traceback lines" if err else "no error output")

    # 6. lab code missing and download blocked
    print("\nRun 6: lab code missing and download blocked")
    nb6 = nbformat.read(str(nb_path), as_version=4)
    nb6.cells = nb6.cells[:3]
    nb6.cells[2].source = (nb6.cells[2].source.replace("https://raw.githubusercontent.com/", "http://127.0.0.1:9/")
                           .replace("https://cdn.jsdelivr.net/gh/", "http://127.0.0.1:9/"))
    execute(nb6, fresh_dir(with_module=False, with_data=False), allow_errors=True)
    s = outputs_text(nb6.cells[2])
    check("missing lab code gives a one-line message with the fallback link",
          "Could not download the lab files" in s and "github.com/almokhtar8-stack/nebula-lab" in s, s.strip()[:120])

    # 7. damaged data file on disk is detected and replaced (here: download blocked, so it stops cleanly)
    print("\nRun 7: damaged data file")
    wd = fresh_dir()
    with open(wd / "data" / "nebula_blood_atac_rna.csv.gz", "r+b") as fh:
        fh.seek(5000)
        fh.write(b"\x00" * 64)
    (wd / "nebula_lab.py").write_text(mod)
    nb7 = nbformat.read(str(nb_path), as_version=4)
    nb7.cells = nb7.cells[:3]
    execute(nb7, wd, allow_errors=True)
    s = outputs_text(nb7.cells[2])
    check("a damaged data file is never used", "could not be downloaded" in s, "checksum caught it")

    out = ROOT / "test" / f"results_{env_label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(RESULTS, indent=2))
    n_fail = sum(not c["pass"] for c in RESULTS["checks"])
    print(f"\n{len(RESULTS['checks']) - n_fail}/{len(RESULTS['checks'])} checks passed. Results: {out}")
    return n_fail


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="colab-matched")
    sys.exit(1 if main(ap.parse_args().env) else 0)
