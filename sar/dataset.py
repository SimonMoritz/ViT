import shutil
from pathlib import Path
import random
from collections import defaultdict


def split_dataset(
    img_dir: Path | str,
    label_dir: Path | str,
    output_dir: Path | str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
):
    """
    Split dataset into train/val/test sets while keeping images from the same airport together.

    Args:
        img_dir: Directory containing images
        label_dir: Directory containing labels
        output_dir: Output directory for split dataset
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.2)
        test_ratio: Proportion of data for testing (default: 0.1)
        seed: Random seed for reproducibility (default: 42)

    Creates directory structure:
        output_dir/
            train/
                images/
                labels/
            val/
                images/
                labels/
            test/
                images/
                labels/

    Note: Images are grouped by airport name (prefix before last underscore and number).
          All images from the same airport will be in the same split.
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_dir = Path(output_dir)

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    # Get all image files
    images = sorted(img_dir.glob("*.jpg"))
    assert images, f"No images found in {img_dir}"

    # Verify all images have corresponding labels
    for img_path in images:
        label_path = label_dir / (img_path.stem + ".txt")
        assert label_path.exists(), f"Missing label for {img_path.name}"

    # Group images by airport
    # Format: American_LosAngeles_1.jpg -> American_LosAngeles
    airport_groups = defaultdict(list)
    for img_path in images:
        # Extract airport name (everything before the last underscore + number)
        parts = img_path.stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            airport_name = parts[0]
        else:
            airport_name = img_path.stem

        airport_groups[airport_name].append(img_path)

    # Get list of airports and shuffle them
    airports = sorted(airport_groups.keys())
    random.seed(seed)
    random.shuffle(airports)

    # Calculate split indices based on number of airports
    n_airports = len(airports)
    n_train = int(n_airports * train_ratio)
    n_val = int(n_airports * val_ratio)

    # Split airports into train/val/test
    train_airports = airports[:n_train]
    val_airports = airports[n_train:n_train + n_val]
    test_airports = airports[n_train + n_val:]

    # Collect images for each split
    splits = {
        "train": [img for airport in train_airports for img in airport_groups[airport]],
        "val": [img for airport in val_airports for img in airport_groups[airport]],
        "test": [img for airport in test_airports for img in airport_groups[airport]],
    }

    # Create directory structure and copy files
    for split_name, split_images in splits.items():
        split_img_dir = output_dir / split_name / "images"
        split_lbl_dir = output_dir / split_name / "labels"

        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            label_path = label_dir / (img_path.stem + ".txt")

            # Copy image and label
            shutil.copy2(img_path, split_img_dir / img_path.name)
            shutil.copy2(label_path, split_lbl_dir / label_path.name)

        # Count unique airports in this split
        split_airports = set()
        for img_path in split_images:
            parts = img_path.stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                split_airports.add(parts[0])
            else:
                split_airports.add(img_path.stem)

        print(f"{split_name:5s}: {len(split_images):3d} images from {len(split_airports):3d} airports")

    print(f"\nDataset split complete! Output: {output_dir}")
    print(f"Total airports: {n_airports}")
