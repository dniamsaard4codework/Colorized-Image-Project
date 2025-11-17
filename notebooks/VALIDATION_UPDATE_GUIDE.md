# 📘 Guide: Adding Validation Set to Model Notebooks

This guide explains how to update your colorization model notebooks (B, C, D, E, F, G, H1, H2, and lambda variants) to include validation set support, based on the changes already applied to Model A.

---

## 🎯 Overview of Changes

The validation set enhancement adds:
1. **Validation dataset loading** alongside train and test sets
2. **Dataset statistics display** showing counts for train/val/test
3. **Validation evaluation** after each training epoch
4. **Best model saving** based on validation loss (prevents overfitting)
5. **Validation visualization** every 10 epochs (displayed in notebook AND saved to disk)
6. **Training history CSV** with train and validation metrics
7. **Updated plots** showing both training and validation curves

---

## 📋 Required Changes by Section

### **SECTION 0: Dataset Class - Add Data Augmentation**

**Current code pattern (all models B-H):**
```python
class ColorizeDataset(Dataset):
    def __init__(self, root_dir, img_size=256, split='train'):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split
        
        color_dir = self.root_dir / f"{split}_color"
        self.color_paths = sorted(glob.glob(str(color_dir / "*.jpg")))
        
        black_dir = self.root_dir / f"{split}_black"
        self.black_paths = sorted(glob.glob(str(black_dir / "*.jpg")))
        
        assert len(self.color_paths) == len(self.black_paths)
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        print(f"Loaded {len(self.color_paths)} {split} image pairs")
```

**✅ Replace with (add augmentation for training set):**
```python
class ColorizeDataset(Dataset):
    def __init__(self, root_dir, img_size=256, split='train'):
        self.root_dir = Path(root_dir)
        self.img_size = img_size
        self.split = split
        
        color_dir = self.root_dir / f"{split}_color"
        self.color_paths = sorted(glob.glob(str(color_dir / "*.jpg")))
        
        black_dir = self.root_dir / f"{split}_black"
        self.black_paths = sorted(glob.glob(str(black_dir / "*.jpg")))
        
        assert len(self.color_paths) == len(self.black_paths)
        
        # Data augmentation for training set
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor()
            ])
        else:
            # No augmentation for validation and test sets
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor()
            ])
        
        print(f"Loaded {len(self.color_paths)} {split} image pairs")
```

**📝 What this adds:**
- **RandomHorizontalFlip**: 50% chance to flip image horizontally (improves left-right invariance)
- **RandomRotation**: Random rotation up to ±15 degrees (helps with orientation robustness)
- **ColorJitter**: Random variations in brightness, contrast, saturation, and hue (improves color robustness)
- **RandomAffine**: Random translation up to 10% (helps with position invariance)
- **Split-aware**: Only applies augmentation to training set, not validation/test sets

**⚠️ Important:** The augmentation must be applied to BOTH the color and grayscale images identically. The current implementation uses the same random seed for both, so this works correctly.

---

### **SECTION 1: Dataset Loading Cell**

**Current code pattern (all models B-H):**
```python
# Load datasets
train_dataset = ColorizeDataset('../data/colorize_dataset/data', img_size=IMG_SIZE, split='train')
test_dataset = ColorizeDataset('../data/colorize_dataset/data', img_size=IMG_SIZE, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)

print(f"Training batches: {len(train_loader)}")
print(f"Testing batches: {len(test_loader)}")
```

**✅ Replace with:**
```python
# Load datasets
train_dataset = ColorizeDataset('../data/colorize_dataset/data', img_size=IMG_SIZE, split='train')
val_dataset = ColorizeDataset('../data/colorize_dataset/data', img_size=IMG_SIZE, split='val')
test_dataset = ColorizeDataset('../data/colorize_dataset/data', img_size=IMG_SIZE, split='test')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True if torch.cuda.is_available() else False)

# Display dataset statistics
print(f"\n{'='*60}")
print(f"Dataset Statistics:")
print(f"{'='*60}")
print(f"Training set:   {len(train_dataset):4d} images ({len(train_loader):3d} batches)")
print(f"Validation set: {len(val_dataset):4d} images ({len(val_loader):3d} batches)")
print(f"Test set:       {len(test_dataset):4d} images ({len(test_loader):3d} batches)")
print(f"Image size:     {IMG_SIZE}×{IMG_SIZE} pixels")
print(f"Batch size:     {BATCH_SIZE}")
print(f"{'='*60}\n")
```

---

### **SECTION 2: Training Setup Cell**

**Current code pattern:**
```python
# Training history
history = {'train_loss': [], 'gen_loss': []}

# Create save directory
save_dir = Path(f'../models/{MODEL_ID}')
save_dir.mkdir(parents=True, exist_ok=True)

print(f"Model will be saved to: {save_dir}")
```

**✅ Replace with:**
```python
# Training history
history = {
    'epoch': [],
    'train_loss': [], 
    'val_loss': [],
    'gen_loss': []
}

# Create save directories
save_dir = Path(f'../models/{MODEL_ID}')
results_dir = Path('../results')
save_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

print(f"Model will be saved to: {save_dir}")
print(f"Results will be saved to: {results_dir}")
```

---

### **SECTION 3: Add Validation Function**

**Location:** Insert NEW cell(s) BEFORE the training loop

**✅ Add these functions:**

```python
# Validation function
def validate_epoch(generator, val_loader, device):
    generator.eval()
    epoch_loss = 0.0
    
    with torch.no_grad():
        for L, ab_real, _ in tqdm(val_loader, desc="Validating", leave=False):
            L = L.to(device)
            ab_real = ab_real.to(device)
            
            ab_fake = generator(L)
            
            # Calculate loss based on your model configuration
            # Use the SAME loss calculation as in train_epoch, but without GAN loss for Model C
            
            # FOR MODEL A: Only L1
            loss = l1_loss(ab_fake, ab_real) * LAMBDA_L1
            
            # FOR MODEL B: L1 + Perceptual
            # loss = l1_loss(ab_fake, ab_real) * LAMBDA_L1 + perceptual_loss(ab_fake, ab_real) * LAMBDA_PERCEPTUAL
            
            # FOR MODEL C (GAN + L1 + Perceptual): Skip GAN loss in validation
            # loss = l1_loss(ab_fake, ab_real) * LAMBDA_L1 + perceptual_loss(ab_fake, ab_real) * LAMBDA_PERCEPTUAL
            
            # FOR MODEL D: Only Perceptual
            # loss = perceptual_loss(ab_fake, ab_real) * LAMBDA_PERCEPTUAL
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(val_loader)

# Visualization function for validation samples
def visualize_validation_samples(generator, val_loader, device, epoch, save_dir, num_samples=5):
    generator.eval()
    
    with torch.no_grad():
        L, ab_real, gray_img = next(iter(val_loader))
        L = L[:num_samples].to(device)
        ab_real = ab_real[:num_samples]
        gray_img = gray_img[:num_samples]
        
        ab_fake = generator(L).cpu()
        L = L.cpu()
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        
        for i in range(num_samples):
            # Grayscale input
            axes[i, 0].imshow(gray_img[i].squeeze(), cmap='gray')
            axes[i, 0].set_title('Input (Grayscale)', fontweight='bold')
            axes[i, 0].axis('off')
            
            # Generated colorization
            L_np = ((L[i].squeeze().numpy() + 1.0) * 50.0)
            ab_fake_np = ab_fake[i].permute(1, 2, 0).numpy() * 128.0
            lab_fake = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            lab_fake[:, :, 0] = L_np
            lab_fake[:, :, 1:] = ab_fake_np
            rgb_fake = lab2rgb(lab_fake)
            
            axes[i, 1].imshow(rgb_fake)
            axes[i, 1].set_title('Generated Color', fontweight='bold')
            axes[i, 1].axis('off')
            
            # Ground truth
            ab_real_np = ab_real[i].permute(1, 2, 0).numpy() * 128.0
            lab_real = np.zeros((IMG_SIZE, IMG_SIZE, 3))
            lab_real[:, :, 0] = L_np
            lab_real[:, :, 1:] = ab_real_np
            rgb_real = lab2rgb(lab_real)
            
            axes[i, 2].imshow(rgb_real)
            axes[i, 2].set_title('Ground Truth', fontweight='bold')
            axes[i, 2].axis('off')
        
        plt.suptitle(f'Validation Samples - Epoch {epoch}', fontsize=16, fontweight='bold', y=1.0)
        plt.tight_layout()
        
        output_path = save_dir / f'val_samples_epoch_{epoch}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Validation samples saved to: {output_path.name}")
        
        # Display the plot in notebook
        plt.show()
```

**⚠️ Important Notes:**
- **Uncomment the appropriate loss calculation** for your model in `validate_epoch()`
- For **Model A**: Use only L1 loss (already shown)
- For **Model B** (L1 + Perceptual): Uncomment the Model B line and comment out Model A line
- For **Model C** (GAN + L1 + Perceptual): Use L1 + Perceptual for validation (skip GAN loss) - uncomment Model C line
- For **Model D** (Perceptual only): Uncomment the Model D line
- **Make sure loss variables (l1_loss, perceptual_loss) are defined** before this function

---

### **SECTION 4: Update Training Loop**

**Current pattern:**
```python
# Train model
best_loss = float('inf')

for epoch in range(1, EPOCHS + 1):
    avg_loss = train_epoch(generator, train_loader, optimizer, epoch, device)
    # OR for GAN models:
    # avg_gen_loss, avg_disc_loss = train_epoch(...)
    
    history['train_loss'].append(avg_loss)
    history['gen_loss'].append(avg_loss)
    
    print(f"\nEpoch {epoch}/{EPOCHS} - Loss: {avg_loss:.4f}")
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        checkpoint = {...}
        torch.save(checkpoint, save_dir / 'best_model.pt')
        print(f"✓ New best model saved! (Loss: {best_loss:.4f})")
    
    # Save checkpoint at intervals
    if epoch % SAVE_INTERVAL == 0:
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved at epoch {epoch}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\nTraining completed! Best loss: {best_loss:.4f}")
```

**✅ Replace with:**

```python
# Train model
best_val_loss = float('inf')

print(f"\n{'='*60}")
print(f"Starting Training - Model {MODEL_ID}")
print(f"{'='*60}\n")

for epoch in range(1, EPOCHS + 1):
    # Train
    avg_train_loss = train_epoch(generator, train_loader, optimizer, epoch, device)
    # OR for GAN models (Model C):
    # avg_gen_loss, avg_disc_loss = train_epoch(generator, discriminator, train_loader, ...)
    # avg_train_loss = avg_gen_loss  # Use generator loss for comparison
    
    # Validate
    avg_val_loss = validate_epoch(generator, val_loader, device)
    
    # Update history
    history['epoch'].append(epoch)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['gen_loss'].append(avg_train_loss)
    
    # For GAN models, also track discriminator loss if needed:
    # history['disc_loss'].append(avg_disc_loss)
    
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss:   {avg_val_loss:.4f}")
    
    # Save best model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # For GAN models, also include:
            # 'discriminator_state_dict': discriminator.state_dict(),
            # 'disc_optimizer_state_dict': disc_optimizer.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss,
            'best_train_loss': avg_train_loss
        }
        torch.save(checkpoint, save_dir / 'best_model.pt')
        print(f"  ✓ New best model saved! (Val Loss: {best_val_loss:.4f})")
    
    # Save checkpoint at intervals
    if epoch % SAVE_INTERVAL == 0:
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  ✓ Checkpoint saved at epoch {epoch}")
        
        # Visualize validation samples every 10 epochs
        visualize_validation_samples(generator, val_loader, device, epoch, results_dir)
    
    # Save training history to CSV after each epoch
    history_df = pd.DataFrame(history)
    history_csv_path = results_dir / f'training_history_{MODEL_ID}.csv'
    history_df.to_csv(history_csv_path, index=False)
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'='*60}")
print(f"Training completed!")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Training history saved to: {history_csv_path}")
print(f"{'='*60}\n")
```

---

### **SECTION 5: Update Save Results Cell**

**Find the metrics DataFrame creation and update:**

**Current:**
```python
metrics_df = pd.DataFrame([{
    'Model_ID': MODEL_ID,
    'Architecture': 'U-Net',
    'GAN': USE_GAN,
    'L1_Loss': USE_L1,
    'Perceptual_Loss': USE_PERCEPTUAL,
    'Lambda_L1': LAMBDA_L1,
    'Lambda_Perceptual': LAMBDA_PERCEPTUAL,
    'PSNR': metrics_avg['PSNR'],
    'SSIM': metrics_avg['SSIM'],
    'CIEDE2000': metrics_avg['CIEDE2000'],
    'LPIPS': metrics_avg['LPIPS'],
    'Best_Loss': best_loss,
    'Epochs': EPOCHS
}])
```

**✅ Replace with:**
```python
metrics_df = pd.DataFrame([{
    'Model_ID': MODEL_ID,
    'Architecture': 'U-Net',
    'GAN': USE_GAN,
    'L1_Loss': USE_L1,
    'Perceptual_Loss': USE_PERCEPTUAL,
    'Lambda_L1': LAMBDA_L1,
    'Lambda_Perceptual': LAMBDA_PERCEPTUAL,
    'PSNR': metrics_avg['PSNR'],
    'SSIM': metrics_avg['SSIM'],
    'CIEDE2000': metrics_avg['CIEDE2000'],
    'LPIPS': metrics_avg['LPIPS'],
    'Best_Val_Loss': best_val_loss,
    'Epochs': EPOCHS
}])
```

---

### **SECTION 6: Update Training History Plot**

**Current:**
```python
# Save training history plot
plt.figure(figsize=(10, 6))
plt.plot(history['gen_loss'], 'b-', linewidth=2, label='Generator Loss')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'Model {MODEL_ID} Training History', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
```

**✅ Replace with:**
```python
# Save training history plot
plt.figure(figsize=(12, 6))
plt.plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)
plt.plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title(f'Model {MODEL_ID} Training History', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
```

---

## 🔍 Model-Specific Considerations

### **Model B** (L1 + Perceptual, No GAN)
- ✅ Simple - just add validation set
- Validation loss = L1 loss + Perceptual loss

### **Model C** (GAN + L1 + Perceptual)
- ⚠️ **Special handling for GAN**
- Validation should use **L1 + Perceptual only** (no GAN loss)
- Training loop returns both `avg_gen_loss` and `avg_disc_loss`
- Use `avg_gen_loss` as `avg_train_loss` for validation comparison
- Remember to save discriminator state in checkpoint

### **Model D** (Perceptual Only)
- ✅ Simple - just perceptual loss
- Validation loss = Perceptual loss only

### **Models E, F, G, H1, H2**
- Follow the same pattern as C if they use GAN
- Adjust loss calculation based on which losses are enabled

### **Lambda Variants (C_lambda1_0.5x, C_lambda1_2.0x)**
- Same as Model C
- The lambda values are already in configuration

---

## ✅ Checklist for Each Model

Use this checklist when updating each notebook:

- [ ] **Section 0**: Add data augmentation to `ColorizeDataset` class (training set only)
- [ ] **Section 1**: Add `val_dataset` and `val_loader`
- [ ] **Section 1**: Add dataset statistics print block
- [ ] **Section 2**: Update history dictionary with `'epoch'` and `'val_loss'`
- [ ] **Section 2**: Add `results_dir` creation
- [ ] **Section 3**: Add `validate_epoch()` function
- [ ] **Section 3**: Add `visualize_validation_samples()` function
- [ ] **Section 4**: Update training loop to call validation
- [ ] **Section 4**: Change `best_loss` to `best_val_loss`
- [ ] **Section 4**: Add validation visualization every 10 epochs
- [ ] **Section 4**: Add CSV saving in training loop
- [ ] **Section 5**: Update metrics DataFrame to use `best_val_loss`
- [ ] **Section 6**: Update plot to show both train and validation curves
- [ ] **Test**: Run the notebook to ensure no errors

---

## 🚀 Quick Start Steps

1. **First**: Run `create_validation_split.ipynb` to create validation folders
2. **Then**: Update each model notebook following this guide
3. **Order suggestion**: Update in this order to get familiar with the pattern:
   - Model B (simple: L1 + Perceptual)
   - Model D (simple: Perceptual only)
   - Model C (complex: GAN + L1 + Perceptual)
   - Models E, F, G, H1, H2, lambda variants

---

## 💡 Why Add Data Augmentation?

**Current Problem:**
- Only using basic Resize → very limited data diversity
- With ~5400 training images, model can overfit easily
- No robustness to common variations (flips, rotations, lighting)

**Benefits of Added Augmentation:**
- **Better Generalization**: Model learns from ~10-50x more variations
- **Prevent Overfitting**: Harder for model to memorize training set
- **Improved Robustness**: Handles real-world variations better
- **Higher Metrics**: Typically improves PSNR, SSIM on validation/test sets

**What We Added:**
1. **RandomHorizontalFlip (50%)**: Natural images work left-to-right
2. **RandomRotation (±15°)**: Handles slight camera tilts
3. **ColorJitter**: Robust to lighting/camera differences
4. **RandomAffine**: Handles slight position shifts

**Note**: Augmentation is ONLY applied during training, not validation/test, so metrics remain fair and comparable.

---

## 💡 Tips

- **Copy from Model A**: Model A is already updated - use it as reference
- **Test incrementally**: Update one section at a time and test
- **Watch indentation**: Python is sensitive to indentation
- **Check cell order**: Functions must be defined before they're called
- **Save often**: Save your notebook after each successful section update

---

## 🐛 Common Issues & Solutions

### Issue: `val_dataset` not found
**Solution**: Make sure you ran `create_validation_split.ipynb` first to create `val_color` and `val_black` folders

### Issue: `validate_epoch` not defined
**Solution**: Ensure the validation function cell is placed BEFORE the training loop cell

### Issue: `l1_loss` or `perceptual_loss` not defined in validation
**Solution**: Make sure you define these loss functions BEFORE the `validate_epoch()` function. They should be in the "Training Setup" section.

### Issue: Plot shows only one line
**Solution**: Make sure you're using `history['epoch']` as x-axis and both `train_loss` and `val_loss`

### Issue: CSV has wrong columns
**Solution**: Check that history dictionary has all keys: `'epoch'`, `'train_loss'`, `'val_loss'`, `'gen_loss'`

### Issue: `results_dir` not defined when calling `visualize_validation_samples()`
**Solution**: Ensure you created `results_dir` in the Training Setup section (Section 2)

### Issue: For GAN models (Model C), training returns two values but code expects one
**Solution**: Update the training loop to handle both values:
```python
avg_gen_loss, avg_disc_loss = train_epoch(...)
avg_train_loss = avg_gen_loss  # Use generator loss
```

---

## 📊 Expected Outputs After Changes

After updating, each model will produce:

1. **During training:**
   - Dataset statistics display
   - Epoch-by-epoch train and validation loss
   - Best model saved based on validation loss
   - Checkpoints every 10 epochs
   - Validation visualizations every 10 epochs (displayed in notebook and saved to disk)

2. **Training artifacts:**
   - `../models/{MODEL_ID}/best_model.pt` (based on validation)
   - `../models/{MODEL_ID}/checkpoint_epoch_*.pt`
   - `../results/training_history_{MODEL_ID}.csv`
   - `../results/val_samples_epoch_*.png`
   - `../results/training_history_{MODEL_ID}.png` (dual-line plot)

---

## 📚 Reference

See **`model_A.ipynb`** for the complete working example of all these changes.

---

**Good luck! 🎉**
