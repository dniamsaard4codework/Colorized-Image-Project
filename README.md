# Colorized-Image-Project

| ID           | Category        | Model name / description                      | GAN? | L1? | Perceptual? | λ₁ (relative)   | λ₂ (relative)   | 
|--------------|-----------------|-----------------------------------------------|------|-----|-------------|-----------------|-----------------|
| A            | Core baseline   | U-Net + L1 (pure regression)                  | No   | Yes | No          | λ₁ = λ₁*        | λ₂ = 0          | 
| B            | Core baseline   | cGAN + L1 (pix2pix-style)                     | Yes  | Yes | No          | λ₁ = λ₁*        | λ₂ = 0          | 
| C            | Core full model | cGAN + L1 + Perceptual (main proposed model)  | Yes  | Yes | Yes         | λ₁ = λ₁*        | λ₂ = λ₂*        | 
| D            | Loss ablation   | U-Net + Perceptual (no L1, no GAN)            | No   | No  | Yes         | λ₁ = 0          | λ₂ = λ₂*        | 
| E            | Loss ablation   | U-Net + L1 + Perceptual (no GAN)              | No   | Yes | Yes         | λ₁ = λ₁*        | λ₂ = λ₂*        | 
| F            | Loss ablation   | cGAN + Perceptual (no L1)                     | Yes  | No  | Yes         | λ₁ = 0          | λ₂ = λ₂*        | 
| C-λ₁-0.5×    | λ₁ ablation     | cGAN + 0.5·L1 + Perceptual                    | Yes  | Yes | Yes         | λ₁ = 0.5·λ₁*    | λ₂ = λ₂*        | 
| C-λ₁-2.0×    | λ₁ ablation     | cGAN + 2.0·L1 + Perceptual                    | Yes  | Yes | Yes         | λ₁ = 2.0·λ₁*    | λ₂ = λ₂*        | 
| C-λ₂-0.5×    | λ₂ ablation     | cGAN + L1 + 0.5·Perceptual                    | Yes  | Yes | Yes         | λ₁ = λ₁*        | λ₂ = 0.5·λ₂*    | 
| C-λ₂-2.0×    | λ₂ ablation     | cGAN + L1 + 2.0·Perceptual                    | Yes  | Yes | Yes         | λ₁ = λ₁*        | λ₂ = 2.0·λ₂*    | 
