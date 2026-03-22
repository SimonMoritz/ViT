from sar import split_dataset
from sar.config import (
    DATASET_SPLIT_SEED,
    RAW_IMAGE_DIR,
    RAW_LABEL_DIR,
    SPLIT_OUTPUT_DIR,
    TEST_SPLIT_RATIO,
    TRAIN_SPLIT_RATIO,
    VAL_SPLIT_RATIO,
)


def main() -> None:
    split_dataset(
        img_dir=RAW_IMAGE_DIR,
        label_dir=RAW_LABEL_DIR,
        output_dir=SPLIT_OUTPUT_DIR,
        train_ratio=TRAIN_SPLIT_RATIO,
        val_ratio=VAL_SPLIT_RATIO,
        test_ratio=TEST_SPLIT_RATIO,
        seed=DATASET_SPLIT_SEED,
    )


if __name__ == "__main__":
    main()
