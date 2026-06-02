import shutil
import urllib.request
import zipfile
from pathlib import Path

from src.utils.config import load_config

DATA_URL = "https://www.manythings.org/anki/fra-eng.zip"


def download_data(data_path):
    dest = Path(data_path)
    if dest.exists():
        print(f"Data already exists at {dest}, skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    zip_path = dest.parent / "fra-eng.zip"

    print(f"Downloading {DATA_URL} ...")
    with urllib.request.urlopen(DATA_URL) as response, open(zip_path, "wb") as f:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 8192
        while True:
            buf = response.read(chunk)
            if not buf:
                break
            f.write(buf)
            downloaded += len(buf)
            if total:
                pct = downloaded / total * 100
                print(f"\r  {pct:.1f}%  ({downloaded:,} / {total:,} bytes)", end="", flush=True)
    print()

    print(f"Extracting to {dest.parent} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith("fra.txt"):
                with zf.open(member) as src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                break
    zip_path.unlink()

    lines = sum(1 for _ in open(dest, encoding="utf-8"))
    print(f"Done. {dest} — {lines:,} sentence pairs.")


def main():
    data_cfg = load_config("configs/data_config.yaml")
    download_data(data_cfg["data_path"])


if __name__ == "__main__":
    main()
