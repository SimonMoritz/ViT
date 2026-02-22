from sar import split_dataset
from sar.config import DEFAULT_IMG_DIR, DEFAULT_LABEL_DIR, DEFAULT_DATASET_OUTPUT_DIR


def main():
    split_dataset(
        img_dir=DEFAULT_IMG_DIR,
        label_dir=DEFAULT_LABEL_DIR,
        output_dir=DEFAULT_DATASET_OUTPUT_DIR,
        seed=42,
    )


if __name__ == "__main__":
    main()
