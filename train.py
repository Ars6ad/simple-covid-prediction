"""
Training Script for COVID-19 CT Scan Classification
Author: AI Coding Assistant
Model Architecture: Simple 3-Layer Convolutional Neural Network (CNN)

Features:
- Loads images via KaggleHub (azaemon/preprocessed-ct-scans-for-covid19)
- Automatically detects classes (COVID Positive, Normal, Non-Informative)
- Stratified 70% Train, 15% Validation, 15% Test split
- Uses TensorFlow tf.data pipeline with batching (memory-safe on laptops)
- Trains CNN with EarlyStopping
- Saves model to models/covid_ct_cnn.keras and class mapping to models/class_mapping.json
- Saves accuracy and loss plots to results/
- Exports test split metadata to results/test_split.json for evaluate.py
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split

# TensorFlow & Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# Set random seed for full reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)
np.random.seed(SEED)

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
DEFAULT_EPOCHS = 15

def get_dataset_dir():
    """Download or locate cached dataset using kagglehub."""
    import kagglehub
    print("\n[Step 1/5] Fetching dataset via KaggleHub...")
    path = kagglehub.dataset_download("azaemon/preprocessed-ct-scans-for-covid19")
    print(f"Dataset path: {path}")
    return path

def collect_images_and_labels(dataset_dir, max_per_class=None):
    """
    Scans dataset directory, identifies class folders, maps them to standard labels:
    - COVID Positive (from 'pCT', 'positive', 'covid')
    - Normal (from 'nCT', 'negative', 'normal')
    - Non-Informative (from 'NiCT', 'non-informative')
    """
    print("\n[Step 2/5] Scanning and preparing dataset...")
    supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    
    # Map folder names to clean display labels
    class_name_map = {
        "pct": "COVID Positive",
        "positive": "COVID Positive",
        "covid": "COVID Positive",
        "covid-19": "COVID Positive",
        "nct": "Normal",
        "negative": "Normal",
        "normal": "Normal",
        "non-covid": "Normal",
        "nict": "Non-Informative",
        "non-informative": "Non-Informative",
        "non_informative": "Non-Informative"
    }

    raw_classes = {}
    target_dir = os.path.join(dataset_dir, "Preprocessed CT scans")
    if not os.path.exists(target_dir):
        target_dir = dataset_dir
    else:
        print(f"Selecting preprocessed scans folder: {target_dir}")

    for root, dirs, files in os.walk(target_dir):
        valid_files = [
            os.path.join(root, f) for f in files
            if Path(f).suffix.lower() in supported_extensions
        ]
        if valid_files:
            folder_name = os.path.basename(root)
            raw_classes[folder_name] = valid_files

    if not raw_classes:
        raise FileNotFoundError(f"No valid images found in dataset directory: {dataset_dir}")

    # Map to standardized classes
    standard_class_images = {}
    for folder, filepaths in raw_classes.items():
        clean_name = class_name_map.get(folder.lower(), folder)
        if clean_name not in standard_class_images:
            standard_class_images[clean_name] = []
        standard_class_images[clean_name].extend(filepaths)

    # Class list in sorted order for consistent indexing
    class_names = sorted(list(standard_class_images.keys()))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = {idx: name for idx, name in enumerate(class_names)}

    all_filepaths = []
    all_labels = []

    print("\nDiscovered Class Distribution:")
    for cls_name in class_names:
        files = standard_class_images[cls_name]
        # Avoid duplicate images if duplicates exist by filename
        unique_files = list(dict.fromkeys(files))
        if max_per_class and len(unique_files) > max_per_class:
            np.random.seed(SEED)
            selected_indices = np.random.choice(len(unique_files), max_per_class, replace=False)
            unique_files = [unique_files[i] for i in selected_indices]
        
        print(f" - {cls_name} (Class {class_to_idx[cls_name]}): {len(unique_files)} images")
        for f in unique_files:
            all_filepaths.append(f)
            all_labels.append(class_to_idx[cls_name])

    all_filepaths = np.array(all_filepaths)
    all_labels = np.array(all_labels)

    return all_filepaths, all_labels, class_names, idx_to_class

def create_tf_dataset(filepaths, labels, is_training=True):
    """
    Creates a high-performance tf.data.Dataset pipeline:
    - Loads raw image bytes
    - Decodes to RGB
    - Resizes to (128, 128)
    - Normalizes to [0, 1]
    - Batches and prefetches for optimal memory usage
    """
    def process_path(file_path, label):
        img_raw = tf.io.read_file(file_path)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0  # Normalize to [0, 1]
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(filepaths), 2000), seed=SEED)
    
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def build_cnn_model(num_classes):
    """
    Builds the CNN architecture specified in the requirements:
    - Input: 128x128x3
    - Conv2D (32, 3x3, ReLU) + MaxPool2D (2x2)
    - Conv2D (64, 3x3, ReLU) + MaxPool2D (2x2)
    - Conv2D (128, 3x3, ReLU) + MaxPool2D (2x2)
    - Flatten
    - Dense (128, ReLU)
    - Dropout (0.3)
    - Output: num_classes (Softmax)
    """
    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        
        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Classification Head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def plot_and_save_history(history, output_dir="results"):
    """Plot and save training/validation accuracy and loss graphs."""
    os.makedirs(output_dir, exist_ok=True)
    
    epochs_range = range(1, len(history.history["loss"]) + 1)
    
    # Accuracy plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["accuracy"], "b-o", label="Training Accuracy")
    plt.plot(epochs_range, history.history["val_accuracy"], "g--s", label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    acc_path = os.path.join(output_dir, "accuracy_plot.png")
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"Accuracy plot saved to: {acc_path}")

    # Loss plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history.history["loss"], "r-o", label="Training Loss")
    plt.plot(epochs_range, history.history["val_loss"], "m--s", label="Validation Loss")
    plt.title("Training vs Validation Loss", fontsize=14, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "loss_plot.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"Loss plot saved to: {loss_path}")

def train(epochs=DEFAULT_EPOCHS, max_per_class=None):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # 1. Dataset location
    dataset_dir = get_dataset_dir()

    # 2. Collect files and labels
    filepaths, labels, class_names, idx_to_class = collect_images_and_labels(
        dataset_dir, max_per_class=max_per_class
    )
    num_classes = len(class_names)
    print(f"\nTotal samples selected for split: {len(filepaths)}")

    # 3. Stratified 70% Train, 15% Validation, 15% Test split
    print("\n[Step 3/5] Splitting dataset into 70% Train, 15% Val, 15% Test (Stratified)...")
    train_files, temp_files, train_lbls, temp_lbls = train_test_split(
        filepaths, labels, test_size=0.30, stratify=labels, random_state=SEED
    )
    val_files, test_files, val_lbls, test_lbls = train_test_split(
        temp_files, temp_lbls, test_size=0.50, stratify=temp_lbls, random_state=SEED
    )

    print(f"  - Training samples:   {len(train_files)} ({len(train_files)/len(filepaths)*100:.1f}%)")
    print(f"  - Validation samples: {len(val_files)} ({len(val_files)/len(filepaths)*100:.1f}%)")
    print(f"  - Testing samples:    {len(test_files)} ({len(test_files)/len(filepaths)*100:.1f}%)")

    # Save test set filepaths and labels for evaluate.py
    test_split_info = {
        "filepaths": test_files.tolist(),
        "labels": test_lbls.tolist(),
        "class_names": class_names,
        "idx_to_class": idx_to_class
    }
    with open("results/test_split.json", "w") as f:
        json.dump(test_split_info, f, indent=2)
    print("Saved test split metadata to: results/test_split.json")

    # Save class mapping
    class_mapping_path = "models/class_mapping.json"
    with open(class_mapping_path, "w") as f:
        json.dump({"idx_to_class": idx_to_class, "class_names": class_names}, f, indent=2)
    print(f"Saved class mapping to: {class_mapping_path}")

    # 4. Create tf.data datasets
    train_ds = create_tf_dataset(train_files, train_lbls, is_training=True)
    val_ds = create_tf_dataset(val_files, val_lbls, is_training=False)

    # 5. Build CNN Model
    print("\n[Step 4/5] Building CNN Architecture...")
    model = build_cnn_model(num_classes)
    model.summary()

    # Callbacks
    model_save_path = "models/covid_ct_cnn.keras"
    early_stop = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # 6. Train Model
    print("\n[Step 5/5] Training the CNN Model...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    # Make sure the final best weights are saved
    model.save(model_save_path)
    print(f"\nTrained model successfully saved to: {model_save_path}")

    # Generate and save plots
    plot_and_save_history(history, output_dir="results")

    print("\nTraining completed successfully! Next step: run 'python evaluate.py'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on COVID-19 CT Scan Dataset")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Optional maximum number of images per class (useful for quick laptop training)")
    args = parser.parse_args()

    train(epochs=args.epochs, max_per_class=args.max_per_class)
