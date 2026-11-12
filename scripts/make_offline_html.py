"""Turn the completed notebook into one HTML file that opens offline in any browser.

    python scripts/make_offline_html.py [out.html]

The published Figure 1c panel is referenced by URL inside the notebook; here it is embedded so the
page needs no internet. (The notebook itself never embeds data.)
"""
import base64
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import nebula_lab as lab  # noqa: E402

FIG_URL = f"https://raw.githubusercontent.com/{lab.REPO}/{lab.REF}/assets/fu2025_fig1c.png"


def main(out):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "-m", "nbconvert", "--to", "html", "--template", "classic",
                        str(ROOT / "NEBULA_Lab_completed.ipynb"), "--output", "page", "--output-dir", tmp],
                       check=True, capture_output=True)
        page = (Path(tmp) / "page.html").read_text(encoding="utf-8")
    data_uri = "data:image/png;base64," + base64.b64encode((ROOT / "assets" / "fu2025_fig1c.png").read_bytes()).decode()
    assert FIG_URL in page, "figure URL not found in the export"
    page = page.replace(FIG_URL, data_uri).replace("<title>NEBULA_Lab_completed</title>",
                                                   "<title>NEBULA LAB, completed run (offline copy)</title>")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "backup" / "NEBULA_Lab_completed_offline.html")
