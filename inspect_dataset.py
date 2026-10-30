"""
Dataset Inspection Script for COVID-19 CT Scan Classification
Author: AI Coding Assistant
Dataset: azaemon/preprocessed-ct-scans-for-covid19 via KaggleHub

Objective:
- Download (or locate cached) dataset from KaggleHub
- Inspect directory layout, class folders, and image formats
- Verify presence and counts of COVID Positive, Normal (or Negative), and Non-Informative classes
- Save a visual sample grid of images to results/sample_images.png
"""

import os
import sys
import glob
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def get_dataset_path():
    """Download or retrieve the local path to the dataset using kagglehub."""
    print("=" * 70)
    print("1. DOWNLOADING / RESOLVING DATASET VIA KAGGLEHUB")
    print("=" * 70)
    try:
        import kagglehub
        print("Contacting KaggleHub for 'azaemon/preprocessed-ct-scans-for-covid19'...")
        path = kagglehub.dataset_download("azaemon/preprocessed-ct-scans-for-covid19")
        print(f"Dataset root directory: {path}")
        return path
    except Exception as e:
        print(f"\n[ERROR] Failed to download or access dataset via KaggleHub: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have an active internet connection.")
        print("2. If Kaggle requires credentials, ensure ~/.kaggle/kaggle.json exists or")
        print("   set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        print("3. Check if kagglehub is installed: pip install kagglehub")
        sys.exit(1)

def find_image_dirs(root_dir):
    """
    Search recursively for directories containing image files.
    Prefers 'Preprocessed CT scans' if present to avoid duplicating with 'Original CT Scans'.
    Returns a dict mapping class/folder name to list of image paths.
    """
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    class_images = {}

    target_dir = os.path.join(root_dir, "Preprocessed CT scans")
    if not os.path.exists(target_dir):
        target_dir = root_dir
    else:
        print(f"[INFO] Using target directory: {target_dir}")

    for root, dirs, files in os.walk(target_dir):
        image_files = [
            os.path.join(root, f) for f in files
            if Path(f).suffix.lower() in supported_extensions
        ]
        if image_files:
            folder_name = os.path.basename(root)
            class_images[folder_name] = image_files

    return class_images

def inspect_dataset(dataset_dir):
    """Inspect dataset folders, file counts, formats, and dimensions."""
    print("\n" + "=" * 70)
    print("2. DATASET DIRECTORY STRUCTURE AND CLASS DISTRIBUTION")
    print("=" * 70)
    print(f"Scanning files in: {dataset_dir}\n")

    # List direct items under root
    items = os.listdir(dataset_dir)
    print(f"Top-level items in dataset folder ({len(items)} items):")
    for item in items[:15]:
        item_path = os.path.join(dataset_dir, item)
        item_type = "DIR" if os.path.isdir(item_path) else "FILE"
        print(f" - [{item_type}] {item}")
    if len(items) > 15:
        print(f"   ... and {len(items) - 15} more items.")

    # Find all directories that contain images
    class_images = find_image_dirs(dataset_dir)
    
    if not class_images:
        print("\n[WARNING] No images found with standard extensions (.jpg, .png, .jpeg).")
        return None

    print("\nDetected image categories / folders:")
    total_images = 0
    class_stats = {}

    for cls_name, img_list in class_images.items():
        count = len(img_list)
        total_images += count
        
        # Inspect sample images for format, size, and mode
        sample_formats = set()
        sample_sizes = set()
        sample_modes = set()
        for img_path in img_list[:10]:
            try:
                with Image.open(img_path) as img:
                    sample_formats.add(img.format)
                    sample_sizes.add(img.size)
                    sample_modes.add(img.mode)
            except Exception:
                pass

        class_stats[cls_name] = {
            "count": count,
            "formats": list(sample_formats),
            "sizes": list(sample_sizes),
            "modes": list(sample_modes),
            "samples": img_list[:3]
        }

        print(f"\n* Class Folder: '{cls_name}'")
        print(f"  - Total Images: {count}")
        print(f"  - Formats: {list(sample_formats)}")
        print(f"  - Image Dimensions (sample): {list(sample_sizes)[:3]}")
        print(f"  - Color Modes: {list(sample_modes)}")
        print(f"  - Sample Image: {img_list[0]}")

    print("\n" + "-" * 70)
    print(f"Total Images across all detected classes: {total_images}")
    print("-" * 70)

    # Class analysis against objective:
    # 1. COVID Positive
    # 2. Normal / Negative
    # 3. Non-Informative
    print("\n" + "=" * 70)
    print("3. CLASS MAPPING VERIFICATION")
    print("=" * 70)
    
    # Check standard naming in this dataset:
    # Commonly:
    # 'pCT' -> Positive CT (COVID Positive)
    # 'nCT' -> Negative CT (Normal / Non-COVID)
    # 'NiCT' -> Non-informative CT
    # Or 'COVID', 'Normal', 'Non-Informative'
    mapping_detected = {}
    for cls in class_stats.keys():
        lower_cls = cls.lower()
        if lower_cls in ["pct", "positive", "covid", "covid-19", "covid_positive"]:
            mapping_detected[cls] = "COVID Positive"
        elif lower_cls in ["nct", "negative", "normal", "non-covid"]:
            mapping_detected[cls] = "Normal"
        elif lower_cls in ["nict", "non-informative", "non_informative", "uninformative"]:
            mapping_detected[cls] = "Non-Informative"
        else:
            mapping_detected[cls] = cls  # Keep as-is if custom

    print("Class mapping to project categories:")
    for orig_cls, mapped in mapping_detected.items():
        print(f"  Folder '{orig_cls}' -> Mapped Class: '{mapped}' ({class_stats[orig_cls]['count']} images)")

    # Save visual sample figure
    os.makedirs("results", exist_ok=True)
    visualize_samples(class_stats, mapping_detected)

    return class_images, mapping_detected

def visualize_samples(class_stats, mapping_detected):
    """Save a sample grid showing images from each class into results/."""
    try:
        classes = list(class_stats.keys())
        n_classes = len(classes)
        samples_per_class = 3

        fig, axes = plt.subplots(n_classes, samples_per_class, figsize=(10, 3 * n_classes))
        fig.suptitle("Sample CT Scan Images per Class", fontsize=14, fontweight="bold")

        if n_classes == 1:
            axes = [axes]

        for i, cls in enumerate(classes):
            sample_paths = class_stats[cls]["samples"][:samples_per_class]
            label_display = mapping_detected.get(cls, cls)
            for j in range(samples_per_class):
                ax = axes[i][j] if n_classes > 1 and samples_per_class > 1 else (axes[j] if n_classes == 1 else axes[i])
                if j < len(sample_paths):
                    img = Image.open(sample_paths[j])
                    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
                    ax.set_title(f"{label_display}\n({os.path.basename(sample_paths[j])})", fontsize=9)
                ax.axis("off")

        plt.tight_layout()
        output_plot = os.path.join("results", "sample_images.png")
        plt.savefig(output_plot, dpi=150)
        plt.close()
        print(f"\n[INFO] Sample images visualization saved to: {output_plot}")
    except Exception as e:
        print(f"[NOTE] Could not save sample visualization: {e}")

def main():
    dataset_path = get_dataset_path()
    inspect_dataset(dataset_path)
    print("\nDataset inspection finished successfully.")

if __name__ == "__main__":
    main()
