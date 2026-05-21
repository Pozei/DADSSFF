# DADSSFF: Domain Alignment Dynamic Spectral and Spatial Feature Fusion for Hyperspectral Change Detection

![Flowchart](Flowchart.png)

Official PyTorch implementation of **DADSSFF**, a novel hyperspectral change detection (HSIs-CD) method published in **IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS), Vol. 18, 2025**.


> **Paper**: [IEEE JSTARS 2025](https://doi.org/10.1109/JSTARS.2024.3495217)

---

## 📖 Abstract

Change detection is an important task in geospatial analysis that aims to identify noticeable variations in geographic elements between images captured at different periods. However, existing methods often overlook the distribution discrepancies across images caused by changes in imaging time. Meanwhile, the spectral and spatial features of hyperspectral images still have great potential for further development in extracting and detecting changes. To mitigate these challenges, we propose DADSSFF for hyperspectral change detection. Key innovations include:

1. **Domain Alignment**: Aligns the mean (first-order statistics) and correlation (second-order statistics) of bitemporal images to alleviate inconsistent feature distributions.
2. **KLD-Enhanced Attention**: Employs Kullback-Leibler divergence to increase interaction between auxiliary networks (spectral & spatial attention branches) and the main network.
3. **Dynamic Feature Fusion**: Uses cosine similarity to adaptively measure the importance of spectral and spatial features.

---

## 🏗️ Network Architecture

DADSSFF consists of **one main network** and **two auxiliary branches**:

```
                    ┌─────────────────────────┐
  T1 ──────────────►│                         │
                    │   Siamese CNN (Shared)  │──► T1_fea, T2_fea
  T2 ──────────────►│                         │
                    └─────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
   Domain Alignment    Difference Feature    Concatenated Feature
   (Mean + CORAL)      │                     │
         │              ▼                     ▼
   loss_DA        Channel Attention     Position Attention
                    (CAM)                 (PAM)
                       │                     │
                       ▼                     ▼
                  loss_fea_img          loss_con_fea
                  (KL Divergence)       (KL Divergence)
                       │                     │
                       └──────┬──────────────┘
                              │
                              ▼
                    Cosine Similarity →
                    Dynamic Weighted Fusion
                              │
                              ▼
                          Classifier
```

## 📁 Repository Structure

```
DADSSFF/
│
├── README.md                          # This file
├── Flowchart.png                      # Network architecture diagram
├── Domain_Alignment_Dynamic_...pdf    # Published paper
│
├── Code/
│   ├── Model_DADSSFF.py               # Core DADSSFF model definition
│   ├── Attention_Module.py            # CAM, PAM, CoAM attention modules (based on DANet)
│   ├── Gain_batch.py                  # Data loading, patch extraction, train/test split
│   ├── GeneratePic.py                 # Visualization & result map generation
│   │
│   ├── Demo_China_main.ipynb          # Training & evaluation on China dataset
│   ├── Demo_River_main.ipynb          # Training & evaluation on River dataset
│   ├── Demo_USA_main.ipynb            # Training & evaluation on USA dataset
│   │
│   └── result/                        # Output prediction maps
│       ├── China_GT.png / China_DADSSFF.png
│       ├── River_GT.png / River_DADSSFF.png
│       └── USA_GT.png   / USA_DADSSFF.png
│
└── Data/
    ├── 01_China/                      # China hyperspectral dataset
    │   ├── China_T1.mat               #   Time-1 HSI
    │   ├── China_T2.mat               #   Time-2 HSI
    │   └── China_GT.mat               #   Ground truth
    ├── 02_River/                      # River hyperspectral dataset
    │   ├── River_T1.mat
    │   ├── River_T2.mat
    │   └── River_GT.mat
    └── 03_USA/                        # USA hyperspectral dataset
        ├── USA_T1.mat
        ├── USA_T2.mat
        └── USA_GT.mat
```

---

## 🚀 Quick Start

### Requirements

- Python 3.7+
- PyTorch 1.8+
- CUDA (recommended)

```bash
pip install torch numpy scipy scikit-learn matplotlib jupyter
```

### Training & Evaluation

Run the Jupyter notebooks for each dataset:

```bash
cd Code/
jupyter notebook Demo_China_main.ipynb   # China dataset
jupyter notebook Demo_River_main.ipynb   # River dataset
jupyter notebook Demo_USA_main.ipynb     # USA dataset
```

**Key hyperparameters** (configurable in notebooks):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--patches`  | 7       | Patch size |
| `--batches`  | 64      | Batch size |
| `--epoches`  | 100     | Training epochs |
| `--tr_rate`  | 0.05    | Training sample ratio (5%) |
| `--lr_rate`  | 0.001   | Learning rate |
| `--decay`    | 0.001   | Weight decay |
| `--lambdas1` | 0.1     | Domain alignment loss weight |
| `--lambdas2` | 0.2     | KL divergence loss weight |
| `--seed`     | 1024    | Random seed |

### Loss Function

The total loss is a combination of three terms:

```python
Total_Loss = CrossEntropyLoss + λ₁ · Loss_DA + λ₂ · Loss_KLD
```

- **Loss_DA**: Domain alignment loss = Mean alignment + CORAL alignment
- **Loss_KLD**: KL divergence between main network and auxiliary branches

---

## 📊 Experimental Results

DADSSFF was evaluated on three public hyperspectral change detection datasets and compared against state-of-the-art methods including CVA, DPCA, SFA, IR-MAD, GETNET, SiamCRNN, ML-EDAN, D²AGCN, SSA-SiamNet, HyperNet, MSDFFN, HyGSTAN, etc.

| Dataset | F1 | Kappa | OA | Precision | Recall |
|---------|--------|---------|--------|---------|--------|---------|
| **China** | 96.41% | 0.9495 | 97.92 | 0.9650 | 0.9633 |
| **River** | 83.74% | 0.8218 | 97.12 | 0.8282 | 0.8468 |
| **USA**   | 93.83% | 0.9203 | 97.21 | 0.9354 | 0.9412 |

DADSSFF achieves **optimal results on F1, OA, and Kappa** across all three datasets, demonstrating superior change detection capability with significantly reduced pseudo-change recognition compared to conventional and deep learning methods.

---

## 📚 Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@article{qin2024domain,
  title={Domain Alignment Dynamic Spectral and Spatial Feature Fusion for Hyperspectral Change Detection},
  author={Qin, Xuexiang and Zhang, Yuxiang and Dong, Yanni},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2025},
  volume={18},
  pages={557--568},
  publisher={IEEE},
  doi={10.1109/JSTARS.2024.3495217}
}
```

---

## 🙏 Acknowledgements
The attention modules (CAM/PAM) are based on [DANet](https://github.com/junfu1115/DANet/) by Fu et al. (CVPR 2019). The CORAL loss implementation is referenced from [DeepCORAL](https://github.com/jindongwang/transferlearning/tree/master/code/DeepDA) by Sun & Saenko (ECCV 2016).

---


