# Image Colorization using Deep Learning
## Comparative Study of U-Net and Conditional GAN with Configurable Loss Functions

---

## 👥 Team Members

**Asian Institute of Technology (AIT)**

1. **Dechathon Niamsa-ard** - st126235
2. **Aphisit Jaemyaem** - st126130

---

## 📋 Project Overview

This project implements a comprehensive deep learning framework for automatic image colorization, comparing multiple architectures and loss function combinations. Given a grayscale input image, the system predicts plausible chrominance information to reconstruct a full-color image in the CIE Lab color space.

### 🎯 Objective

Learn a mapping function:

$$G_\theta : \mathbb{R}^{1 \times H \times W} \to \mathbb{R}^{2 \times H \times W}$$

where:
- **Input**: Luminance channel (L) from CIE Lab color space
- **Output**: Chrominance channels (a, b) in normalized form

---

## 🏗️ Project Structure

```
Colorized-Image-Project/
│
├── data/
│   └── colorize_dataset/
│       └── data/
│           ├── train_color/    # 4,750 RGB training images
│           ├── train_black/    # 4,750 grayscale training images
│           ├── val_color/      # 250 RGB validation images
│           ├── val_black/      # 250 grayscale validation images
│           ├── test_color/     # 739 RGB test images
│           └── test_black/     # 739 grayscale test images
│
├── models/                     # Saved model checkpoints
│   ├── A/                      # Model A checkpoints
│   ├── B/                      # Model B checkpoints
│   ├── C/                      # Model C checkpoints
│   ├── D/                      # Model D checkpoints
│   ├── E/                      # Model E checkpoints
│   ├── F/                      # Model F checkpoints
│   ├── C_L1_LOW/              # Ablation study checkpoints
│   ├── C_L1_HIGH/
│   ├── C_PERCEPTUAL_LOW/
│   └── C_PERCEPTUAL_HIGH/
│
├── results/                    # Training history, metrics, and visualizations
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── D/
│   ├── E/
│   ├── F/
│   ├── C_L1_LOW/
│   ├── C_L1_HIGH/
│   ├── C_PERCEPTUAL_LOW/
│   ├── C_PERCEPTUAL_HIGH/
│   └── Evaluate/              # Comprehensive evaluation results
│
├── notebooks/
│   ├── model_A.ipynb          # U-Net + L1 (pure regression)
│   ├── model_B.ipynb          # cGAN + L1 (pix2pix-style)
│   ├── model_C.ipynb          # cGAN + L1 + Perceptual (main model)
│   ├── model_D.ipynb          # U-Net + Perceptual
│   ├── model_E.ipynb          # U-Net + L1 + Perceptual
│   ├── model_F.ipynb          # cGAN + Perceptual
│   ├── model_C_L1_LOW.ipynb   # Ablation: Low L1 weight
│   ├── model_C_L1_HIGH.ipynb  # Ablation: High L1 weight
│   ├── model_C_PERCEPTUAL_LOW.ipynb   # Ablation: Low perceptual weight
│   ├── model_C_PERCEPTUAL_HIGH.ipynb  # Ablation: High perceptual weight
│   ├── comprehensive_model_evaluation.ipynb  # Complete evaluation
│   ├── configurable_experiment.ipynb         # Configurable template
│   └── create_validation_split.ipynb         # Dataset preparation
│
├── scripts/
│   └── device_config_test.py  # Hardware configuration testing
│
├── main.py                    # Main training script
├── pyproject.toml            # Project dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 🧪 Model Configurations

### Core Models (A–F)

| Model ID | Architecture | GAN | L1 Loss | Perceptual Loss | Description |
|----------|-------------|-----|---------|-----------------|-------------|
| **A** | U-Net | ❌ | ✅ | ❌ | Pure regression baseline (λ₁=100) |
| **B** | U-Net + PatchGAN | ✅ | ✅ | ❌ | Pix2pix-style (λ₁=100) |
| **C** | U-Net + PatchGAN | ✅ | ✅ | ✅ | **Main proposed model** (λ₁=100, λ₂=10) |
| **D** | U-Net | ❌ | ❌ | ✅ | Perceptual only (λ₂=10) |
| **E** | U-Net | ❌ | ✅ | ✅ | No adversarial (λ₁=100, λ₂=10) |
| **F** | U-Net + PatchGAN | ✅ | ❌ | ✅ | No L1 loss (λ₂=10) |

### Ablation Studies (Model C Variants)

| Model ID | λ₁ (L1) | λ₂ (Perceptual) | Description |
|----------|---------|-----------------|-------------|
| **C_L1_LOW** | 50.0 | 10.0 | Lower L1 weight (0.5×) |
| **C_L1_HIGH** | 200.0 | 10.0 | Higher L1 weight (2.0×) |
| **C_PERCEPTUAL_LOW** | 100.0 | 5.0 | Lower perceptual weight (0.5×) |
| **C_PERCEPTUAL_HIGH** | 100.0 | 20.0 | Higher perceptual weight (2.0×) |

### Loss Function Formulation

**Generator Objective:**

$$\mathcal{L}_G = \lambda_1 \cdot \mathcal{L}_{L1} + \lambda_2 \cdot \mathcal{L}_{perc} + \mathcal{L}_{adv}$$

Where:
- $\mathcal{L}_{L1}$: Pixel-wise L1 loss on chrominance channels
- $\mathcal{L}_{perc}$: VGG16-based perceptual loss (extracted from relu1_2, relu2_2, relu3_3)
- $\mathcal{L}_{adv}$: Adversarial loss from PatchGAN discriminator

---

## 🏛️ Architecture Details

### U-Net Generator

- **Encoder**: 7 downsampling blocks (64 → 512 channels)
- **Bottleneck**: Compressed representation at 1×1 spatial resolution
- **Decoder**: 7 upsampling blocks with skip connections
- **Output**: 2-channel chrominance prediction with Tanh activation
- **Total Parameters**: ~54.4M

### PatchGAN Discriminator

- **Architecture**: 5 convolutional blocks (3 → 512 channels)
- **Output**: Patch-wise classification map (16×16)
- **Total Parameters**: ~2.8M
- **Loss Function**: Binary cross-entropy with logits

### VGG16 Perceptual Loss

- **Pretrained**: ImageNet weights (frozen)
- **Feature Extraction Layers**: relu1_2, relu2_2, relu3_3
- **Loss Computation**: L1 distance in feature space

---

## 📊 Dataset

### Data Source

**Kaggle Dataset**: [Image Colorization Dataset](https://www.kaggle.com/datasets/aayush9753/image-colorization-dataset)

### Dataset Statistics

- **Total Images**: 5,739 image pairs (color + grayscale)
- **Training Set**: 4,750 images (82.8%)
- **Validation Set**: 250 images (4.4%)
- **Test Set**: 739 images (12.8%)
- **Resolution**: 256×256 pixels
- **Color Space**: CIE Lab

### Data Augmentation (Training Only)

- Horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random affine transformation (translation ±10%)

---

## 🎓 Training Configuration

### Hyperparameters

- **Batch Size**: 64
- **Epochs**: 50
- **Learning Rate**: 2×10⁻⁴
- **Optimizer**: Adam (β₁=0.5, β₂=0.999)
- **LR Schedule**: Linear decay to zero after epoch 25
- **Hardware**: NVIDIA GeForce RTX 5060 Ti

### Training Process

1. **Discriminator Update** (if GAN enabled):
   - Real pairs: (L, ab_real) → label 1
   - Fake pairs: (L, G(L)) → label 0
   - Loss: (ℒ_real + ℒ_fake) / 2

2. **Generator Update**:
   - Compute composite loss: ℒ_L1 + ℒ_adv + ℒ_perc
   - Backpropagate through U-Net

3. **Validation**:
   - L1 loss on validation set
   - Save best model based on validation loss

---

## 📈 Evaluation Metrics

### Quantitative Metrics

1. **PSNR (Peak Signal-to-Noise Ratio)**
   - Measures reconstruction quality in dB
   - Higher is better
   - Computed on RGB images

2. **SSIM (Structural Similarity Index)**
   - Measures structural similarity (0-1)
   - Higher is better
   - Considers luminance, contrast, and structure

3. **CIEDE2000 (ΔE₀₀)**
   - Perceptual color difference in Lab space
   - Lower is better
   - Sampled at 8-pixel stride

4. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - Deep learning-based perceptual similarity
   - Lower is better
   - AlexNet backbone

---

## 🏆 Results

### Overall Performance Comparison

| Model | PSNR (dB) ↑ | SSIM ↑ | CIEDE2000 ↓ | LPIPS ↓ | Val Loss ↓ | Overall Score |
|-------|-------------|--------|-------------|---------|------------|---------------|
| **A** | **23.71** | 0.9229 | **10.17** | **0.2016** | **0.0725** | **0.9992** |
| **E** | 23.68 | **0.9235** | 10.26 | 0.2094 | 0.0735 | 0.9749 |
| **C** | 21.05 | 0.8560 | 13.41 | 0.2490 | 0.0787 | 0.4909 |
| D | 21.51 | 0.7101 | 12.40 | 0.2112 | 0.0893 | 0.4876 |
| C_PERCEPTUAL_HIGH | 20.84 | 0.8529 | 13.62 | 0.2432 | 0.0827 | 0.4787 |
| C_PERCEPTUAL_LOW | 20.87 | 0.8420 | 13.15 | 0.2524 | 0.0783 | 0.4675 |
| **B** | 20.83 | 0.8298 | 13.75 | 0.2715 | 0.0825 | 0.3756 |
| C_L1_LOW | 20.23 | 0.8303 | 14.37 | 0.2526 | 0.0821 | 0.3572 |
| C_L1_HIGH | 20.42 | 0.8369 | 14.20 | 0.2643 | 0.0815 | 0.3551 |
| F | 19.55 | 0.8112 | 15.41 | 0.3041 | 0.0939 | 0.1184 |

### Visual Results Comparison

Below is a comprehensive comparison grid showing how each model performs on the same set of test images:

![Complete Model Comparison Grid](results/Evaluate/complete_comparison_grid.png)

*Figure: Visual comparison of all 10 models (A-F and ablation variants) on selected test images. Each row represents a different test image, with columns showing: Input (grayscale), Ground Truth (original color), and colorization results from each model.*

### Key Findings

#### 🥇 Best Overall Model from Overall Score: **Model A (U-Net + L1)**

- **Overall Score**: 0.9992 (normalized across all metrics)
- **Strengths**:
  - Highest PSNR (23.71 dB)
  - Lowest color difference (CIEDE2000: 10.17)
  - Lowest perceptual distance (LPIPS: 0.2016)
  - Best validation loss (0.0725)
  - Most stable training process
  
#### 🥈 Second Best from Overall Score: **Model E (U-Net + L1 + Perceptual, no GAN)**

- **Overall Score**: 0.9749
- **Strengths**:
  - Highest SSIM (0.9235)
  - Comparable PSNR to Model A (23.68 dB)
  - Better perceptual quality than GAN-based models
  - Stable training without adversarial loss

#### 🥉 Third Best from Overall Score: **Model C (Main Proposed - cGAN + L1 + Perceptual)**

- **Overall Score**: 0.4909
- **Characteristics**:
  - More diverse and creative colorizations
  - Lower quantitative scores but potentially more visually appealing
  - Higher training complexity and instability due to GAN
  - Better suited for creative colorization tasks

### Insights

1. **Simplicity vs. Complexity**: Pure L1 regression (Model A) outperforms complex GAN-based models on all quantitative metrics, suggesting that simpler approaches may be more effective for objective color reconstruction.

2. **GAN Trade-offs**: Adding adversarial loss increases training complexity and reduces metric scores, though it may produce more diverse and creative colorizations that are subjectively more appealing.

3. **Perceptual Loss Impact**: Perceptual loss helps when combined with L1 loss without GAN (Model E), but does not significantly improve GAN-based models.

4. **Ablation Study Results**: Loss weight variations (C_L1_*, C_PERCEPTUAL_*) show marginal differences, suggesting that Model C's default weights (λ₁=100, λ₂=10) are well-balanced.

---

## 🚀 Usage

### Requirements

```bash
# Install dependencies
pip install torch torchvision
pip install numpy pandas matplotlib seaborn
pip install scikit-image Pillow tqdm
pip install lpips colormath torchsummary
```

### Training a Model

1. Open the desired notebook (e.g., `notebooks/model_A.ipynb`)
2. Ensure dataset path is correct: `../data/colorize_dataset/data/`
3. Run all cells sequentially
4. Monitor training progress with tqdm progress bars
5. Checkpoints are saved in `../models/{MODEL_ID}/`
6. Results are saved in `../results/{MODEL_ID}/`

### Evaluation

Run `notebooks/comprehensive_model_evaluation.ipynb` to:
- Load all trained models
- Compare performance on the test set
- Generate side-by-side visualizations
- Create metric comparison plots
- Export evaluation summary

---

## 📁 Output Files

### Per Model

- `models/{MODEL_ID}/best_model.pt` - Best generator checkpoint
- `models/{MODEL_ID}/checkpoint_epoch_*.pt` - Interval checkpoints
- `results/{MODEL_ID}/metrics_{MODEL_ID}.csv` - Test metrics
- `results/{MODEL_ID}/training_history_{MODEL_ID}.csv` - Loss history
- `results/{MODEL_ID}/training_history_{MODEL_ID}.png` - Loss curves
- `results/{MODEL_ID}/colorization_results_{MODEL_ID}.png` - Visual results
- `results/{MODEL_ID}/val_samples_{MODEL_ID}_epoch_*.png` - Validation samples

### Evaluation Summary

- `results/Evaluate/comprehensive_metrics_comparison.png` - Metric bar plots
- `results/Evaluate/training_history_comparison.png` - All training curves
- `results/Evaluate/complete_comparison_grid.png` - All models on same images
- `results/Evaluate/comparison_test_image_*.png` - Per-image comparisons
- `results/Evaluate/inference_metrics_comparison.csv` - Inference statistics

---

## 🔬 Technical Details

### Color Space Processing

**Input Normalization:**

```
L_norm = L*/50.0 - 1.0        # L* ∈ [0,100] → [-1,1]
a_norm = a*/128.0             # a* ∈ [-128,128] → [-1,1]
b_norm = b*/128.0             # b* ∈ [-128,128] → [-1,1]
```

**Output Denormalization:**

```
L* = (L_norm + 1.0) × 50.0
a* = a_norm × 128.0
b* = b_norm × 128.0
```

### Reproducibility

- Fixed random seed: 42
- Deterministic cuDNN operations
- Seeded data loaders
- Same train/val/test split across all experiments

---

## 📝 Notebooks Description

### Training Notebooks

- **model_A.ipynb** to **model_F.ipynb**: Individual model training with detailed explanations and inline documentation
- **configurable_experiment.ipynb**: Template for custom configurations and experiments

### Ablation Studies

- **model_C_L1_LOW/HIGH.ipynb**: L1 loss weight sensitivity analysis
- **model_C_PERCEPTUAL_LOW/HIGH.ipynb**: Perceptual loss weight sensitivity analysis

### Evaluation & Utilities

- **comprehensive_model_evaluation.ipynb**: Complete comparative analysis with visualizations
- **create_validation_split.ipynb**: Dataset preparation and splitting utility

---

## 🎯 Key Contributions

1. **Comprehensive Framework**: Systematic comparison of 10 model variants with consistent evaluation
2. **Rigorous Evaluation**: Multiple complementary metrics (PSNR, SSIM, CIEDE2000, LPIPS)
3. **Ablation Studies**: Thorough analysis of loss weight sensitivity
4. **Reproducible Pipeline**: Complete documentation and structured codebase
5. **Surprising Finding**: Simple L1 regression outperforms complex GAN architectures on quantitative metrics

---

## 📚 References

### Architectures

- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", MICCAI 2015
- **Pix2pix**: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", CVPR 2017
- **PatchGAN**: Markovian discriminator from pix2pix

### Loss Functions

- **Perceptual Loss**: Johnson et al., "Perceptual Losses for Real-Time Style Transfer and Super-Resolution", ECCV 2016
- **LPIPS**: Zhang et al., "The Unreasonable Effectiveness of Deep Features as a Perceptual Metric", CVPR 2018

### Color Space

- **CIE Lab**: Commission Internationale de l'Éclairage (International Commission on Illumination)
- **CIEDE2000**: Sharma et al., "The CIEDE2000 Color-Difference Formula: Implementation Notes", Color Research & Application 2005

---

## 🤝 Acknowledgments

This project was developed as part of a Computer Vision course at the **Asian Institute of Technology (AIT)**.

Special thanks to:
- Course instructors for their guidance and support
- PyTorch and scikit-image communities for excellent tools and documentation
- Original authors of U-Net, pix2pix, and perceptual loss papers

---

## 📄 License

This project is for educational purposes only.

---

## 📧 Contact

For questions or collaboration opportunities:

- **Dechathon Niamsa-ard** - st126235
- **Aphisit Jaemyaem** - st126130

**Asian Institute of Technology (AIT)**

---

**Last Updated**: November 2024
