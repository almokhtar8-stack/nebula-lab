"""Check the published repository over the internet, then, and only then, make the QR code.

    python scripts/verify_and_make_qr.py

Every file a student's Colab session needs is downloaded from GitHub and compared byte for byte
(SHA-256) with the local copy. The QR code, which opens the repository page, is written only if
every check passes, and a dated record of the checks is kept in test/qr_verification.json.
"""
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_URL = "https://github.com/almokhtar8-stack/nebula-lab"
RAW = "https://raw.githubusercontent.com/almokhtar8-stack/nebula-lab/{ref}/"
REFS = ("main", "v1.0")            # the Colab link opens main; the notebook downloads from the v1.0 tag
FILES = ["README.md", "NEBULA_Lab.ipynb", "NEBULA_Lab_completed.ipynb", "nebula_lab.py",
         "data/nebula_blood_atac_rna.csv.gz", "assets/fu2025_fig1c.png"]
INK = "#123E38"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    record = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "repo": REPO_URL, "files": []}
    ok = True
    for ref in REFS:
        for f in FILES:
            local = sha((ROOT / f).read_bytes())
            try:
                remote = sha(urllib.request.urlopen(RAW.format(ref=ref) + f, timeout=60).read())
            except Exception as exc:
                remote = f"unreachable: {type(exc).__name__}"
            good = remote == local
            ok &= good
            record["files"].append({"ref": ref, "file": f, "local_sha256": local, "remote_sha256": remote,
                                    "identical": good})
            print(f"[{'PASS' if good else 'FAIL'}] {ref}: {f}")
    record["all_identical"] = ok
    out = ROOT / "test" / "qr_verification.json"
    out.write_text(json.dumps(record, indent=2))
    if not ok:
        print("\nNot every published file matches. No QR code was made.")
        return 1

    import qrcode
    import qrcode.image.svg
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=30, border=4)
    qr.add_data(REPO_URL)
    qr.make(fit=True)
    qr.make_image(fill_color=INK, back_color="white").save(ROOT / "assets" / "qr_nebula_lab.png")
    qrcode.make(REPO_URL, image_factory=qrcode.image.svg.SvgPathImage,
                error_correction=qrcode.constants.ERROR_CORRECT_M).save(str(ROOT / "assets" / "qr_nebula_lab.svg"))
    try:                                         # read the code back, as a phone camera would
        import cv2
        import numpy as np
        from PIL import Image
        img = np.asarray(Image.open(ROOT / "assets" / "qr_nebula_lab.png").convert("RGB"))[:, :, ::-1]
        decoded = cv2.QRCodeDetector().detectAndDecode(img)[0]
        record["qr_decodes_to"] = decoded
        out.write_text(json.dumps(record, indent=2))
        if decoded != REPO_URL:
            print(f"QR decodes to {decoded!r}, not {REPO_URL}")
            return 1
        print("[PASS] the QR code reads back as", decoded)
    except ImportError:
        print("(OpenCV not installed: QR not read back)")
    print(f"\nAll {len(FILES)} files are live and identical on {' and '.join(REFS)}. QR code written for {REPO_URL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
