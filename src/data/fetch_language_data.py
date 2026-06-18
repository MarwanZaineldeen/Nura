from __future__ import annotations

from pathlib import Path

from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DATASET_NAME = "papluca/language-identification"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME)

    split_paths = {
        "train": DATA_DIR / "lang_train.csv",
        "validation": DATA_DIR / "lang_val.csv",
        "test": DATA_DIR / "lang_test.csv",
    }

    for split, output_path in split_paths.items():
        dataset[split].to_pandas().to_csv(output_path, index=False)
        print(f"Saved {split} split to {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
