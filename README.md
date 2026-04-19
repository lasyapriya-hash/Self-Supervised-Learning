# A Glitch in the Matrix: SSL Robustness & Dynamics

This repository contains the implementation and analysis of Self-Supervised Learning (SSL) paradigms (MoCo, SimCLR, VICReg, Barlow Twins) compared against Supervised Learning (SL) across three research axes.

## 👥 Team Members
* Borra Bhavitha
* Hasini
* Lekhya Pillarikuppam
* M Lasya Priya
* Patcha Jyothika

## 📊 Key Findings
* **Label Noise (Hypothesis 1):** SSL models maintained ~86% accuracy under 50% label noise, while Supervised Learning crashed to 36.88%.
* **Generalization (Hypothesis 2):** SSL "Generalist" (MLM) outperformed Supervised "Specialist" (POS) by 4% on cross-domain NER tasks.
* **Critical Periods (Hypothesis 3):** Early-stage training glitches (t=5) cause irreversible damage, resulting in ~10% lower final accuracy compared to late-stage glitches.

## 🧪 Results Summary (Hypothesis 3)
| Corruption Onset (t) | 20C / 10R Accuracy | 10C / 10R Accuracy | 20C / 20R Accuracy |
| :--- | :--- | :--- | :--- |
| **5 (Early)** | 70.03% | 73.16% | 73.96% |
| **15 (Mid)** | 75.07% | 76.71% | 77.94% |
| **30 (Late)** | 79.30% | 79.28% | 78.20% |
*Baseline Accuracy: 81.43%*

## 📁 How to Use
1. **Full Analysis:** Read `SSL_Project_Report.pdf` for detailed methodology and results.
2. **Environment:** ```bash
pip install torch torchvision transformers numpy
