from pathlib import Path

from src.utils.config import load_config


DATASET_REPO = "findnitai/english-to-hinglish"


def download_data(data_path):
    dest = Path(data_path)
    if dest.exists():
        print(f"Data already exists at {dest}, skipping download.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required to download the Hinglish dataset. "
            "Install it with: pip install datasets"
        ) from exc

    print(f"Downloading {DATASET_REPO} from HuggingFace …")
    ds = load_dataset(DATASET_REPO, split="train")
    print(f"  Downloaded {len(ds):,} rows. Writing to {dest} …")

    with open(dest, "w", encoding="utf-8") as f:
        for row in ds:
            en = " ".join(row["translation"]["en"].split())
            hi_ng = " ".join(row["translation"]["hi_ng"].split())
            # Skip rows where either side is empty
            if not en or not hi_ng:
                continue
            f.write(f"{en}\t{hi_ng}\n")

    lines = sum(1 for _ in open(dest, encoding="utf-8"))
    print(f"Done. {dest} — {lines:,} sentence pairs.")


def main():
    data_cfg = load_config("configs/data_config.yaml")
    download_data(data_cfg["data_path"])


if __name__ == "__main__":
    main()
