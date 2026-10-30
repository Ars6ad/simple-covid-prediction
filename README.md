
# COVID-19 CT Scan Classification using CNN

A beginner-friendly machine learning project that classifies Computed Tomography (CT) scan images into three categories using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed via Streamlit.

---

## 1. Project Objective

The goal is to automatically classify chest CT scan slices into three distinct clinical and quality categories:

1. **COVID Positive (`pCT`):** CT slices demonstrating characteristic imaging signs of COVID-19 pneumonia (e.g. ground-glass opacities, consolidation).
2. **Normal / Negative (`nCT`):** CT slices exhibiting normal lung tissue or pathologies unrelated to COVID-19.
3. **Non-Informative (`NiCT`):** CT slices that lack sufficient lung parenchyma for evaluation (e.g., slices capturing mostly neck, upper bone structures, abdomen, or severe motion artifacts).

---

## 2. Dataset

- **Source:** Kaggle (`azaemon/preprocessed-ct-scans-for-covid19`)
- **Download Method:** Automated retrieval using `kagglehub`.
- **Dataset Structure:**
  - `pCT/`: COVID-19 positive scans (~4,001 images)
  - `nCT/`: Negative / Normal scans (~9,979 images)
  - `NiCT/`: Non-informative scans (~5,705 images)
- **Image Size:** 128 × 128 pixels (3 channels RGB)
- **Split Ratio:** Stratified 70% Training, 15% Validation, 15% Testing.

---

## 3. CNN Model Architecture

A lightweight, 3-layer Convolutional Neural Network designed to run efficiently on standard consumer laptops without requiring a dedicated GPU:

```
Input: (128, 128, 3)
  │
  ├── Conv2D (32 filters, 3×3 kernel, ReLU activation, same padding)
  ├── MaxPooling2D (2×2 pool size)
  │
  ├── Conv2D (64 filters, 3×3 kernel, ReLU activation, same padding)
  ├── MaxPooling2D (2×2 pool size)
  │
  ├── Conv2D (128 filters, 3×3 kernel, ReLU activation, same padding)
  ├── MaxPooling2D (2×2 pool size)
  │
  ├── Flatten
  ├── Dense (128 neurons, ReLU activation)
  ├── Dropout (rate = 0.3)
  └── Dense (3 neurons, Softmax activation)
```

- **Loss Function:** `sparse_categorical_crossentropy`
- **Optimizer:** `Adam` (learning rate = 0.001)
- **Metrics:** `accuracy`
- **Callbacks:** `EarlyStopping` (monitors `val_loss`, patience=4) and `ModelCheckpoint`.

---

## 4. Project File Structure

```
covid-ct-classification/
│
├── app.py                     # Streamlit web application
├── train.py                   # CNN training pipeline with tf.data
├── inspect_dataset.py         # Dataset inspection and distribution script
├── evaluate.py                # Test set evaluation and metric generation
├── requirements.txt           # Project dependencies
├── README.md                  # Detailed documentation and guide
├── .gitignore                 # Git ignore file for models and datasets
│
├── models/
│   ├── covid_ct_cnn.keras     # Saved trained Keras model
│   └── class_mapping.json     # Class index mapping dictionary
│
└── results/
    ├── sample_images.png      # Visual sample of dataset classes
    ├── accuracy_plot.png      # Training vs validation accuracy
    ├── loss_plot.png          # Training vs validation loss
    ├── confusion_matrix.png   # Normalized and raw confusion matrix
    ├── classification_report.txt # Precision, recall, and F1 report
    └── test_split.json        # Test set paths for reproducible evaluation
```

---

## 5. Execution Instructions (Windows / PowerShell)

### Step 1: Create and Activate Virtual Environment

Open PowerShell or VS Code terminal in the project folder:

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

*(If PowerShell script execution is restricted, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### Step 2: Install Dependencies

```powershell
pip install -r requirements.txt
```

### Step 3: Inspect the Dataset

Download and analyze the dataset structure, image counts, and formats:

```powershell
python inspect_dataset.py
```

### Step 4: Train the CNN Model

Train the model on the stratified dataset (70% Train, 15% Val, 15% Test):

```powershell
python train.py
```

*Tip for quick testing on laptop:* You can optionally limit samples per class using the `--max-per-class` argument:
```powershell
python train.py --epochs 10 --max-per-class 1000
```

### Step 5: Evaluate Model on Test Set

Evaluate the trained model on the isolated 15% test set:

```powershell
python evaluate.py
```

This will print the test accuracy, precision, recall, and F1-score, and save:
- `results/confusion_matrix.png`
- `results/classification_report.txt`

### Step 6: Launch the Streamlit Web App

Start the interactive web application:

```powershell
streamlit run app.py
```

Open your browser at `http://localhost:8501`. You can upload any CT scan image (JPG/PNG) to predict whether it is COVID Positive, Normal, or Non-Informative.

---

## 6. Troubleshooting

- **KaggleHub Authentication:** If Kaggle prompts for credentials, create an API token on [kaggle.com](https://www.kaggle.com) (`Account -> Create New API Token`), and place the downloaded `kaggle.json` into:
  - Windows: `C:\Users\<Username>\.kaggle\kaggle.json`
  - Or set environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- **Memory Optimization:** The training pipeline uses `tf.data` generator streaming to process batches directly from disk, preventing Out-Of-Memory (OOM) errors even with ~20,000 images.

---

## 7. Educational & Medical Disclaimer

> **IMPORTANT:** This project is developed strictly for **educational and academic research purposes** as part of a college machine learning curriculum. It is **not** a certified medical diagnostic system. Predictions must not be used for medical decisions, diagnosis, or treatment.
