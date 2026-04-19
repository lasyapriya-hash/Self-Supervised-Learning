# A Glitch in the Matrix: SSL Robustness & Dynamics

This repository contains the implementation and empirical analysis of Self-Supervised Learning (SSL) paradigms compared against Supervised Learning (SL) across three primary research axes.

## 👥 Team Members
* **Borra Bhavitha G**
* **Hasini Lekhya Pillarikuppam**
* **M Lasya Priya**
* **Patcha Jyothika**

---

## 📊 Hypothesis 1: Robustness to Label Noise
SSL feature extraction provides an inherent defense against corrupted downstream labels.

### Classification Accuracy (%) across Label Noise Levels (CIFAR-10)
| Noise Level | Supervised | SimCLR | MoCo | Barlow Twins | VICReg |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0%** | 72.68 | 76.42 | 77.68 | 86.58 | 86.37 |
| **10%** | 66.27 | 76.30 | 77.61 | 86.41 | 86.29 |
| **30%** | 52.77 | 77.75 | 76.15 | 85.31 | 85.81 |
| **50%** | 36.88 | 76.92 | 75.93 | 85.20 | 86.14 |

---

## 📊 Hypothesis 2: Quality of Latent Representations
SSL pre-training yields superior "generalist" features that generalize better to unseen tasks than specialized supervised models.

### Accuracy (%) Comparison for In-Domain and Cross-Domain Tasks
| Task | Supervised (POS Specialist) | Self-Supervised (MLM Generalist) |
| :--- | :--- | :--- |
| **POS Tagging (In-Domain)** | 87.91 | 84.26 |
| **NER (Cross-Domain)** | 86.60 | 90.40 |

---

## 📊 Hypothesis 3: Sensitivity to Early Training Corruption
We identified a "critical period" in early SSL training where structural data corruption inflicts irreversible damage on representation quality.

### Accuracy (%) across Corruption and Recovery Settings (VICReg)
| Onset Time (t) | 20C / 10R | 10C / 10R | 20C / 20R |
| :--- | :--- | :--- | :--- |
| **5 (Early)** | 70.03 | 73.16 | 73.96 |
| **10** | 70.92 | 75.79 | 75.09 |
| **15** | 75.07 | 76.71 | 77.94 |
| **20** | 77.25 | 78.32 | 78.93 |
| **25** | 78.22 | 78.72 | 79.62 |
| **30 (Late)** | 79.30 | 79.28 | 78.20 |

---

## 📁 Repository Contents
1. **Full Analysis:** Read `SSL_Project_Report.pdf` for detailed methodology and insights.
2. **Environment:** Requirements include `torch`, `torchvision`, `transformers`, and `numpy`.
3. **Execution:** Scripts are organized by hypothesis. All vision experiments utilize a modified ResNet-18 architecture with 3x3 `conv1` and `Identity` maxpooling.
