from sar import split_dataset


def main():
    split_dataset(
        img_dir="Airport_Dataset_v0_images",
        label_dir="Airport_Dataset_v0_labels",
        output_dir="dataset",
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        seed=42,
    )


if __name__ == "__main__":
    main()
