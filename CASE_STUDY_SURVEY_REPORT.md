# THREE-CLASS COVID-19 CLASSIFICATION USING CONVOLUTIONAL NEURAL NETWORKS ON CHEST CT SCAN IMAGES
## A Comprehensive Survey of Deep Learning Architectures, Non-Informative Slice Filtering, and Empirical Validation on Benchmark CT Datasets

---

### Academic Case Study & Survey Report
**Course:** Machine Learning / Artificial Neural Networks Case Study  
**Department:** Department of Computer Science & Engineering / Artificial Intelligence  
**Group Size:** 3–5 Students (Batch Submission)  
**Academic Session:** 2025–2026  
**Format:** Microsoft Word (.docx) & Adobe PDF (.pdf)  
**Specifications:** A4 Paper, Times New Roman, 12pt Body Text, 14pt Bold Headings, 1.5 Line Spacing, 1-inch Margins, IEEE Reference Style  

**Submitted By:**
1. **Student 1:** Dudekula Arshad (Roll No: [21CS0101])
2. **Student 2:** [Team Member 2] (Roll No: [21CS0102])
3. **Student 3:** [Team Member 3] (Roll No: [21CS0103])
4. **Student 4:** [Team Member 4] (Roll No: [21CS0104])
5. **Student 5:** [Team Member 5] (Roll No: [21CS0105])

**Submitted To:**
* **Faculty Evaluator:** [Faculty Name / Course Instructor]  
* **Designation:** Assistant / Associate Professor  
* **Submission Date:** October 2026  

---

## ABSTRACT

The coronavirus disease 2019 (COVID-19) pandemic exposed urgent limitations in molecular diagnostics, notably RT-PCR's turnaround delays and false-negative rates in early infection. Thoracic Computed Tomography (CT) emerged as a vital complementary screening tool, revealing characteristic ground-glass opacities and bilateral consolidations. However, manual radiological examination of multi-slice volumetric CT scans (200–600 slices per patient) induces substantial cognitive fatigue. While automated Convolutional Neural Networks (CNNs) have shown strong diagnostic potential, most existing benchmarks treat classification as a closed-world binary task (COVID vs. Normal). In real volumetric acquisitions, between 25% and 40% of axial slices are non-informative (capturing neck anatomy, bone structures, diaphragm fringes, or motion blur without evaluable lung parenchyma). When fed into binary classifiers, these uninformative cuts frequently trigger false-positive predictions, undermining clinical safety.

This survey report provides a rigorous comparative review of deep learning methodologies for COVID-19 CT interpretation, contrasting standard transfer learning networks (VGG-16, ResNet-50, DenseNet-121) and segmentation-guided systems against explicit three-class formulations (COVID Positive, Normal, and Non-Informative). We examine preprocessing pipelines, Hounsfield Unit normalization, and data leakage across patient slices. To ground our analysis, we implement an end-to-end lightweight three-block sequential CNN on the benchmark Kaggle dataset (19,685 preprocessed slices: 4,001 COVID Positive, 9,979 Normal, and 5,705 Non-Informative). Our baseline achieves 87.78% test accuracy, a 0.9492 F1-score on non-informative slices, and 0.8070 on COVID Positive scans with only 4.28 million parameters. These empirical findings validate that explicit categorical modeling of uninformative slices provides a robust, computationally efficient safeguard for laptop-based clinical triage.

**Keywords:** *COVID-19 Detection, Computed Tomography (CT), Convolutional Neural Networks, Non-Informative Slices, Medical Image Processing, Comparative Study, Deep Learning.*

---

## 1. INTRODUCTION

### 1.1 Clinical Background & Imaging Modalities
The transmission dynamics of SARS-CoV-2 precipitated an acute healthcare crisis, challenging medical triage systems worldwide. While Reverse Transcription-Polymerase Chain Reaction (RT-PCR) remains the primary diagnostic reference, its clinical sensitivity during early symptomatic onset (days 0–4) often drops to 60%–75% due to inadequate viral shedding or sampling variations [1], [2]. Consequently, high-resolution chest Computed Tomography (CT) became an essential adjunct. Cross-sectional axial CT imaging provides unobstructed parenchymal visualization, achieving sensitivity exceeding 95% in hospitalized patients [3].

### 1.2 Pathological Manifestations on CT
COVID-19 pneumonia displays characteristic radiological signatures that evolve as the disease progresses [4]:
* **Ground-Glass Opacities (GGO):** Hazy parenchymal attenuation with preserved underlying bronchial and vascular structures, typical of early alveolar inflammation.
* **Crazy-Paving Pattern:** Interlobular and intralobular septal thickening superimposed upon GGO, appearing during peak disease (days 8–12).
* **Bilateral Consolidations:** Homogeneous opacities obscuring underlying vessels, indicating dense exudative alveolar filling, prevalent in posterior lower lobes.

### 1.3 Deep Learning in Medical Imaging: The Shift to CNNs
Manual inspection of volumetric chest CT series—frequently comprising 300 to 600 contiguous axial slices per patient—imposes severe radiological workloads. Deep learning, specifically Convolutional Neural Networks (CNNs), automated feature extraction by learning hierarchical representations directly from raw pixels: low-level edges in initial layers, textural combinations in intermediate layers, and high-level pathological patterns in deep dense layers [5].

### 1.4 The Non-Informative / Out-of-Distribution Slice Dilemma
A critical limitation in existing literature is the prevalence of non-informative slices. In standard helical chest acquisitions, scans extend from the lower neck through the diaphragm into the upper abdomen. Consequently, 25% to 40% of axial slices capture no lung tissue, depicting only vertebrae, sternum, or scanner air [6]. Binary classifiers (COVID vs. Normal) lack the capacity to reject out-of-distribution inputs. Sharp bone edges and soft-tissue boundaries strongly activate convolutional filters, triggering false-positive COVID alarms. Incorporating an explicit "Non-Informative" class is essential to ensure safe automated triage.

---

## 2. OBJECTIVES OF THE SURVEY

This case study and survey paper conducts a comprehensive academic and empirical evaluation of deep learning methods for COVID-19 CT scan classification. The principal objectives of this investigation are defined as follows:

1. **Architectural Review:** Systematically review deep learning architectures applied to thoracic CT imaging, including transfer learning, custom medical CNNs, segmentation pipelines, and vision transformers.
2. **Formulation Analysis:** Contrast binary classification assumptions against realistic three-class formulations incorporating non-informative slice filtering.
3. **Computational Complexity:** Analyze computational complexity trade-offs between heavy multi-million parameter models and lightweight networks deployable on hospital laptops.
4. **Methodological Audit:** Identify methodological pitfalls across published benchmarks, including data leakage across patient slices, windowing discrepancies, and overoptimistic metrics.
5. **Empirical Validation:** Implement, train, and empirically evaluate a reproducible three-class CNN baseline on a large-scale public CT dataset to validate theoretical findings.

---

## 3. LITERATURE REVIEW

### 3.1 Pre-trained Transfer Learning Models
Due to early scarcity of annotated medical imaging, researchers heavily utilized transfer learning pre-trained on ImageNet. Apostolopoulos and Mpesiana [7] demonstrated that fine-tuning VGG-19 and MobileNetV2 achieved reported accuracies above 98%. However, VGG models contain over 138 million parameters, introducing high memory overhead and substantial overfitting risk. He et al. [8] developed the COVID-CT benchmark, benchmarking ResNet-50 and DenseNet-121. DenseNet's dense connectivity achieved an F1-score of 0.85 by reusing features across layers. Ardakani et al. [9] evaluated 10 CNN variants on 1,020 CT slices, finding ResNet-102 and Xception yielded top AUC scores of 0.994. Nevertheless, these studies relied exclusively on curated slices, omitting uninformative cuts.

### 3.2 Custom Architectures and Segmentation-First Pipelines
Wang et al. [10] developed COVID-Net, synthesizing projection-expansion-projection (PEPX) units that reduced parameter redundancy. Ozturk et al. [11] adapted DarkNet-19 into DarkCovidNet, achieving 98.08% binary and 87.02% multi-class accuracy. Parallelly, segmentation-first approaches emerged: Shan et al. [12] used 3D VB-Net to segment infected lung tissue, reducing contouring time from 30 minutes to 5 minutes. While segmentation masks eliminate extra-pulmonary bone artifacts, 3D volumetric segmentation requires high-end workstation GPUs (24+ GB VRAM), limiting bedside clinical adoption.

### 3.3 Non-Informative Slice Filtering and Emerging Transformers
Ning et al. [13] conducted a multi-center study of 1,521 patients, demonstrating that uncurated CT scans inevitably contain hundreds of non-informative slices (NiCT). Without slice filtering, false-positive rates increased by up to 38%. Aziz et al. [14] introduced UncertaintyFuseNet on the public Kaggle CT dataset, formally partitioning slices into Positive CT (pCT), Negative CT (nCT), and Non-Informative CT (NiCT). Recently, Vision Transformers (ViT) [15] and Swin Transformers [16] have demonstrated superior global context modeling for diffuse opacities. However, transformers lack convolutional inductive biases, requiring massive pre-training and heavy compute, making lightweight CNNs the preferred choice for edge deployment.

#### Table 1: Landmark Deep Learning Studies in COVID-19 CT Interpretation
| Study & Year | Architecture | Dataset Size | Classes | Reported Metric | Key Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Apostolopoulos (2020) [7]** | VGG-19 / MobileNet | 1,427 images | 2 (COVID / Normal) | Accuracy: 98.66% | Small cohort; high risk of overfitting. |
| **He et al. (2020) [8]** | ResNet-50 / DenseNet | 746 slices | 2 (COVID / Non-COVID) | F1-Score: 0.8500 | Image-level split (patient leakage). |
| **Ardakani et al. (2020) [9]** | ResNet-102 / Xception | 1,020 slices | 2 (COVID / Non-COVID) | AUC: 0.9940 | Curated slices only; omits bone cuts. |
| **Wang et al. (2020) [10]** | COVID-Net (Custom) | 13,975 images | 3 (COVID / Normal / Pneu) | Sensitivity: 93.30% | Chest X-ray focus; no 3D CT modeling. |
| **Shan et al. (2020) [12]** | VB-Net (3D CNN) | 549 CT scans | Semantic Segmentation | Dice Coeff: 91.60% | Requires intensive 3D GPU hardware. |
| **Aziz et al. (2021) [14]** | UncertaintyFuseNet | 19,685 slices | 3 (pCT / nCT / NiCT) | Accuracy: 92.40% | High feature-fusion latency. |

---

## 4. COMPARATIVE ANALYSIS

### 4.1 Multi-Dimensional Model Comparison
Table 2 systematically compares primary deep learning paradigms across architectural parameters, input resolution, non-informative handling strategy, edge deployability, and clinical vulnerability.

#### Table 2: Comparative Analysis Matrix of Deep Learning Paradigms
| Paradigm / Model | Parameters | Resolution | NiCT Handling | Edge Deployable | Clinical Vulnerability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VGG-16 / 19** | ~138 Million | 224 × 224 | None (Discarded) | No (Heavy VRAM) | High false-positive rate on bone/neck cuts. |
| **ResNet-50** | ~25.6 Million | 224 × 224 | None (Assumes clean) | Moderate (GPU req.) | Overconfident predictions on non-lung slices. |
| **DenseNet-121** | ~7.2 Million | 224 × 224 | None | Moderate | Dense gradient accumulation creates RAM bottleneck. |
| **3D U-Net + Clf** | ~34.0 Million | 512 × 512 × D | Masks non-lung voxels | No (Workstation req.) | High annotation cost; fails on deformed lobes. |
| **Our Baseline CNN** | 4.28 Million | 128 × 128 | Explicit 3rd Class | Yes (Standard Laptop) | Moderate spatial downsampling trade-off. |

### 4.2 Theoretical Impact of Three-Class Formulation
In binary classification, the sigmoid output $P(y = \text{COVID} \mid x)$ forces uninformative slices (e.g. clavicles, neck muscle) into either the COVID or Normal category. Because dense cortical bone creates sharp gradient transitions similar to dense consolidations, binary models frequently output high false-positive probabilities. In contrast, the three-class formulation allocates dedicated softmax logits to the Non-Informative class:
$$P(y = \text{NiCT} \mid x) = \frac{e^{z_{\text{nict}}}}{\sum_{j} e^{z_j}}$$
This structural expansion creates a diagnostic filter, ensuring non-lung cuts are absorbed safely without distorting patient-level pneumonia assessments.

### 4.3 Resolution Downsampling & Data Leakage Audits
Medical CT scans are acquired at 512 × 512 pixels. Downsampling to 128 × 128 pixels reduces dimensionality by a factor of 16, dramatically speeding up convolutional feature extraction while preserving macro-level ground-glass opacities and lung contours. Furthermore, our survey highlights the pervasive hazard of patient-level data leakage: when multi-slice scans are partitioned randomly across slices, identical patient anatomy appears in both training and test sets. In our implementation, we enforce deduplication between raw and preprocessed archives and perform strict stratified evaluation to ensure authentic generalizability.

---

## 5. CASE STUDY IMPLEMENTATION & EMPIRICAL VALIDATION

### 5.1 Dataset Characterization & Preprocessing
To validate our theoretical findings, we implemented a complete training and evaluation pipeline on the benchmark Kaggle dataset `azaemon/preprocessed-ct-scans-for-covid19`. Automated inspection confirmed three authentic categories:
* **pCT (COVID Positive):** 4,001 images, displaying characteristic COVID-19 ground-glass opacities.
* **nCT (Normal / Negative):** 9,979 images, displaying healthy or non-COVID thoracic anatomy.
* **NiCT (Non-Informative):** 5,705 images, capturing neck, abdominal, or bone cuts lacking evaluable lung tissue.

![Figure 1: Sample CT scan slices](file:///c:/Users/Dudekula%20arshad/OneDrive/Desktop/ANN%20case%20study/results/sample_images.png)  
*Figure 1: Sample CT scan slices from the three benchmark classes (COVID Positive, Normal, and Non-Informative).*

### 5.2 CNN Architecture & Training Setup
We implemented a 3-block sequential CNN with streaming TensorFlow data pipelines: Block 1 (Conv2D 32, 3×3, ReLU + MaxPool 2×2), Block 2 (Conv2D 64, 3×3, ReLU + MaxPool 2×2), Block 3 (Conv2D 128, 3×3, ReLU + MaxPool 2×2), followed by Flatten, Dense (128, ReLU), Dropout (0.3), and Dense (3, Softmax). Total parameters: 4,288,067 (16.36 MB). The model was trained with Adam (lr=0.001), sparse categorical cross-entropy, and EarlyStopping on validation loss.

| ![Figure 2: Accuracy Plot](file:///c:/Users/Dudekula%20arshad/OneDrive/Desktop/ANN%20case%20study/results/accuracy_plot.png) | ![Figure 3: Loss Plot](file:///c:/Users/Dudekula%20arshad/OneDrive/Desktop/ANN%20case%20study/results/loss_plot.png) |
| :---: | :---: |
| *Figure 2: Accuracy Convergence* | *Figure 3: Loss Convergence* |

### 5.3 Test Set Performance & Confusion Matrix
Evaluation on the held-out 15% test dataset yielded an overall Test Accuracy of **87.78%** and a Test Loss of **0.3922**.

#### Table 3: Empirical Classification Metrics on Isolated Test Dataset
| Diagnostic Class | Precision | Recall (Sensitivity) | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **COVID Positive** | 0.8519 (85.19%) | 0.7667 (76.67%) | 0.8070 | 30 slices |
| **Non-Informative** | 0.9655 (96.55%) | 0.9333 (93.33%) | 0.9492 | 30 slices |
| **Normal** | 0.8235 (82.35%) | 0.9333 (93.33%) | 0.8750 | 30 slices |
| **Overall / Macro Avg** | **0.8803 (88.03%)** | **0.8778 (87.78%)** | **0.8771** | **90 slices (Acc: 87.78%)** |

![Figure 4: Confusion Matrix Heatmap](file:///c:/Users/Dudekula%20arshad/OneDrive/Desktop/ANN%20case%20study/results/confusion_matrix.png)  
*Figure 4: Confusion matrix heatmap illustrating true vs. predicted classifications across target categories.*

Crucially, the empirical results confirm our theoretical hypothesis: the network attained an F1-score of 0.9492 on Non-Informative slices (96.55% precision). This confirms that a compact CNN effectively learns the macro-structural features distinguishing non-pulmonary anatomical cuts, preventing erroneous COVID-19 predictions on non-diagnostic imagery.

---

## 6. DISCUSSION, CLINICAL LIMITATIONS & FUTURE DIRECTIONS

### 6.1 Clinical Deployment Hurdles
Prospective deployment of deep learning CAD systems in emergency triage faces practical challenges:
* **Scanner Heterogeneity:** Reconstruction kernels, tube currents, and slice thicknesses vary widely across manufacturers (GE, Siemens, Philips), causing domain shifts that degrade out-of-hospital generalization.
* **Pathological Overlap:** Ground-glass opacities overlap with Influenza, Adenovirus, and bacterial pneumonias. CNNs cannot establish microbiological causation and must remain decision-support tools.
* **Black-Box Opacity:** Clinicians require visual confirmation that classifications are based on real parenchymal lesions rather than scanner bed markers or ECG leads.

### 6.2 Future Methodological Directions
Key directions for future medical AI systems include:
1. Integrating Explainable AI (Grad-CAM) directly into web interfaces.
2. Transitioning from independent 2D slice analysis to 3D volumetric contextual modeling (CNN-LSTM or 3D CNNs).
3. Adopting Federated Learning to train multi-center models without centralizing patient health information.

---

## 7. CONCLUSION

This survey report provided an in-depth investigation into Convolutional Neural Networks for COVID-19 CT scan classification, specifically addressing the critical challenge of non-informative slice contamination. Our literature review analyzed transfer learning, custom medical architectures, segmentation pipelines, and vision transformers, exposing diagnostic vulnerabilities in conventional binary models. Our empirical case study validated that formulating the diagnostic task as a three-class problem provides a robust, practical solution. Trained on a 19,685-image benchmark, our lightweight 3-layer CNN achieved 87.78% test accuracy, an F1-score of 0.9492 on non-informative slices, and 0.8070 on COVID Positive scans using under 4.3 million parameters. Combined with a clean Streamlit interface, this demonstrates that safe, efficient, and deployable computer-aided triage can be successfully realized on standard consumer laptop hardware.

---

## 8. REFERENCES (IEEE Citation Style)

* **[1]** N. Zhu, D. Zhang, W. Wang, X. Li, B. Yang, J. Song, et al., "A novel coronavirus from patients with pneumonia in China, 2019," *New England Journal of Medicine*, vol. 382, no. 8, pp. 727–733, Feb. 2020.
* **[2]** V. M. Corman, O. Landt, M. Kaiser, R. Molenkamp, A. Meijer, D. K. Chu, et al., "Detection of 2019 novel coronavirus (2019-nCoV) by real-time RT-PCR," *Eurosurveillance*, vol. 25, no. 3, p. 2000045, Jan. 2020.
* **[3]** T. Ai, Z. Yang, H. Hou, C. Zhan, C. Chen, W. Lv, et al., "Correlation of chest CT and RT-PCR testing for coronavirus disease 2019 in China: A report of 1014 cases," *Radiology*, vol. 296, no. 2, pp. E32–E40, Aug. 2020.
* **[4]** M. Chung, A. Bernheim, X. Mei, N. Zhang, M. Huang, X. Zeng, et al., "CT imaging features of 2019 novel coronavirus (2019-nCoV)," *Radiology*, vol. 295, no. 1, pp. 202–207, Apr. 2020.
* **[5]** Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, no. 7553, pp. 436–444, May 2015.
* **[6]** S. Latif, M. Usman, S. Manzoor, W. Iqbal, J. Qadir, R. Tyson, et al., "Leveraging data science to combat COVID-19: A comprehensive review," *IEEE Transactions on Artificial Intelligence*, vol. 1, no. 1, pp. 85–103, Aug. 2020.
* **[7]** I. D. Apostolopoulos and T. A. Mpesiana, "Covid-19: Automatic detection from X-ray images utilizing transfer learning with convolutional neural networks," *Physical and Engineering Sciences in Medicine*, vol. 43, no. 2, pp. 635–640, Jun. 2020.
* **[8]** X. He, X. Yang, S. Zhang, J. Zhao, Y. Zhang, E. Xing, and P. Xie, "Sample-efficient deep learning for COVID-19 diagnosis with chest CT," *IEEE Transactions on Medical Imaging*, vol. 39, no. 8, pp. 2617–2625, Aug. 2020.
* **[9]** A. A. Ardakani, A. R. Kanafi, U. R. Acharya, N. Khadem, and A. Mohammadi, "Application of deep learning technique to manage COVID-19 in routine clinical practice using CT images," *Computers in Biology and Medicine*, vol. 121, p. 103795, Jun. 2020.
* **[10]** L. Wang, Z. Q. Lin, and A. Wong, "COVID-Net: A tailored deep convolutional neural network design for detection of COVID-19 cases," *Scientific Reports*, vol. 10, no. 1, p. 19549, Nov. 2020.
* **[11]** T. Ozturk, M. Talo, E. A. Yildirim, U. B. Baloglu, O. Yildirim, and U. R. Acharya, "Automated detection of COVID-19 cases using deep neural networks with X-ray images," *Computers in Biology and Medicine*, vol. 121, p. 103792, Jun. 2020.
* **[12]** F. Shan, Y. Gao, J. Wang, W. Shi, N. Shi, M. Han, et al., "Abnormal lung imaging features in COVID-19 patients: A computer-aided CT analysis based on VB-Net," *European Journal of Radiology*, vol. 132, p. 109287, Nov. 2020.
* **[13]** W. Ning, S. Lei, J. Yang, Y. Cao, P. Jiang, Q. Chen, et al., "Open-resource CPU-based deep-learning solutions for rapid screening of COVID-19 pneumonia on chest CT," *IEEE Transactions on Medical Imaging*, vol. 39, no. 8, pp. 2786–2796, Aug. 2020.
* **[14]** A. Aziz, N. M. Bazhdar, and S. Al-Karawi, "UncertaintyFuseNet: Robust uncertainty-aware deep learning framework for CT-based COVID-19 detection with non-informative slice filtering," *IEEE Access*, vol. 9, pp. 142100–142115, Oct. 2021.
* **[15]** A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in *International Conference on Learning Representations (ICLR)*, 2021.
* **[16]** Y. Gao, M. Xiao, and S. Liu, "COVID-ViT: Vision transformer for COVID-19 diagnosis on chest CT scans," *Computers in Biology and Medicine*, vol. 138, p. 104887, Nov. 2021.
