"""
Evaluation Script for COVID-19 CT Scan Classification
Author: AI Coding Assistant

Requirements:
- Load trained model from models/covid_ct_cnn.keras
- Evaluate on the separate 15% test set saved during training (results/test_split.json)
- Compute and print:
  - Overall test accuracy & test loss
  - Precision, Recall, F1-Score per class
  - Classification report saved to results/classification_report.txt
  - Confusion matrix and heatmap visualization saved to results/confusion_matrix.png
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

def load_test_data(test_info_path="results/test_split.json"):
    """Load test dataset filepaths and true labels saved during train.py."""
    if not os.path.exists(test_info_path):
        print(f"\n[ERROR] Test split file '{test_info_path}' not found!")
        print("Please train the model first by running: python train.py")
        sys.exit(1)
        
    with open(test_info_path, "r") as f:
        data = json.load(f)
        
    filepaths = data["filepaths"]
    labels = np.array(data["labels"])
    class_names = data["class_names"]
    idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
    
    return filepaths, labels, class_names, idx_to_class

def create_eval_dataset(filepaths, labels):
    """Create tf.data evaluation dataset matching training preprocessing."""
    def process_path(file_path, label):
        img_raw = tf.io.read_file(file_path)
        img = tf.io.decode_image(img_raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMAGE_SIZE)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

def plot_confusion_matrix(cm, class_names, output_path="results/confusion_matrix.png"):
    """Plots and saves confusion matrix with counts and percentages."""
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - COVID-19 CT Classification", fontsize=14, fontweight="bold")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=30, ha="right", fontsize=11)
    plt.yticks(tick_marks, class_names, fontsize=11)

    # Label every cell with raw count and row percentage
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / row_sum * 100.0) if row_sum > 0 else 0.0
            text = f"{val}\n({pct:.1f}%)"
            color = "white" if val > thresh else "black"
            plt.text(j, i, text, horizontalalignment="center", verticalalignment="center",
                     color=color, fontsize=10, fontweight="bold")

    plt.ylabel("True Class", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Confusion matrix plot saved to: {output_path}")

def main():
    model_path = "models/covid_ct_cnn.keras"
    if not os.path.exists(model_path):
        print(f"\n[ERROR] Model file '{model_path}' not found!")
        print("Please train the model first by running: python train.py")
        sys.exit(1)

    print("=" * 70)
    print("EVALUATING MODEL ON HELD-OUT TEST DATASET")
    print("=" * 70)

    # 1. Load test metadata
    filepaths, true_labels, class_names, idx_to_class = load_test_data()
    print(f"Loaded {len(filepaths)} test samples across {len(class_names)} classes.")
    
    # Check sample size per class
    for idx, cname in idx_to_class.items():
        count = np.sum(true_labels == idx)
        if count < 10:
            print(f" [NOTE] Class '{cname}' has only {count} test samples; metrics may have higher variance.")

    # 2. Load model
    print(f"\nLoading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 3. Predict on test dataset
    print("Generating predictions on test set...")
    test_ds = create_eval_dataset(filepaths, true_labels)
    
    # Evaluate loss and accuracy
    eval_results = model.evaluate(test_ds, verbose=1)
    test_loss = eval_results[0]
    test_acc = eval_results[1]

    # Predict probabilities
    pred_probs = model.predict(test_ds, verbose=1)
    pred_labels = np.argmax(pred_probs, axis=1)

    # 4. Classification Report
    report = classification_report(
        true_labels,
        pred_labels,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 70)
    print(f"TEST ACCURACY: {test_acc * 100:.2f}% | TEST LOSS: {test_loss:.4f}")
    print("=" * 70)
    print("\nCLASSIFICATION REPORT:")
    print(report)

    # Save classification report to txt
    os.makedirs("results", exist_ok=True)
    report_file = "results/classification_report.txt"
    with open(report_file, "w") as f:
        f.write("COVID-19 CT Scan CNN Classification Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Test Loss:     {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Full classification report saved to: {report_file}")

    # 5. Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(cm, class_names, output_path="results/confusion_matrix.png")

    print("\nEvaluation complete! View graphs in results/ directory.")

if __name__ == "__main__":
    main()
