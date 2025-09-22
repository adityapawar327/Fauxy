#!/usr/bin/env python3
"""
Download balanced dataset with both real and AI-generated faces
"""

import kaggle
import os
import shutil
import zipfile
from pathlib import Path
import random
from PIL import Image
import numpy as np

def download_real_faces():
    """Download real face dataset"""
    print("📥 Downloading real faces dataset...")
    
    # Download a small real faces dataset
    try:
        kaggle.api.dataset_download_files(
            'ashwingupta3012/human-faces', 
            path='dataset/temp_real',
            unzip=True
        )
        print("✅ Real faces dataset downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download real faces: {e}")
        try:
            # Alternative dataset
            kaggle.api.dataset_download_files(
                'atulanandjha/lfwpeople', 
                path='dataset/temp_real',
                unzip=True
            )
            print("✅ Alternative real faces dataset downloaded")
            return True
        except Exception as e2:
            print(f"❌ Failed to download alternative dataset: {e2}")
            return False

def download_ai_faces():
    """Download AI-generated faces dataset"""
    print("📥 Downloading AI-generated faces dataset...")
    
    try:
        kaggle.api.dataset_download_files(
            'shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset',
            path='dataset/temp_ai',
            unzip=True
        )
        print("✅ AI faces dataset downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download AI faces: {e}")
        return False

def collect_images(source_dir, target_dir, label, max_images=2000):
    """Collect images from source directory"""
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(source_path.rglob(f'*{ext}'))
        image_files.extend(source_path.rglob(f'*{ext.upper()}'))
    
    print(f"Found {len(image_files)} {label} images")
    
    # Randomly sample images if we have too many
    if len(image_files) > max_images:
        image_files = random.sample(image_files, max_images)
        print(f"Sampled {max_images} {label} images")
    
    # Copy and process images
    copied = 0
    for i, img_file in enumerate(image_files):
        try:
            # Load and validate image
            with Image.open(img_file) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize to standard size
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                
                # Save with new name
                new_name = f"{label}_{i:04d}.jpg"
                img.save(target_path / new_name, 'JPEG', quality=95)
                copied += 1
                
        except Exception as e:
            print(f"⚠️ Skipped {img_file}: {e}")
            continue
    
    print(f"✅ Processed {copied} {label} images")
    return copied

def organize_dataset():
    """Organize dataset into train/val splits"""
    print("📁 Organizing dataset...")
    
    # Create directory structure
    base_dir = Path("dataset/balanced")
    for split in ['train', 'val']:
        for class_name in ['real', 'fake']:
            (base_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process real images
    real_count = 0
    real_temp_dirs = [
        "dataset/temp_real",
        "dataset/temp_real/human-faces",
        "dataset/temp_real/lfw-deepfunneled",
        "dataset/temp_real/lfw_funneled"
    ]
    
    for temp_dir in real_temp_dirs:
        if Path(temp_dir).exists():
            count = collect_images(temp_dir, "dataset/temp_processed/real", "real", 1500)
            real_count += count
            break
    
    # Process AI images
    ai_temp_dirs = [
        "dataset/temp_ai",
        "dataset/temp_ai/AI-face-detection-Dataset",
        "dataset/temp_ai/dataset"
    ]
    
    ai_count = 0
    for temp_dir in ai_temp_dirs:
        if Path(temp_dir).exists():
            count = collect_images(temp_dir, "dataset/temp_processed/fake", "fake", 1500)
            ai_count += count
            break
    
    print(f"📊 Dataset summary: {real_count} real, {ai_count} fake images")
    
    # Split into train/val
    for class_name in ['real', 'fake']:
        class_dir = Path(f"dataset/temp_processed/{class_name}")
        if not class_dir.exists():
            continue
            
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        
        # 80% train, 20% val
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to final locations
        for img in train_images:
            shutil.copy2(img, base_dir / "train" / class_name / img.name)
        
        for img in val_images:
            shutil.copy2(img, base_dir / "val" / class_name / img.name)
        
        print(f"✅ {class_name}: {len(train_images)} train, {len(val_images)} val")
    
    # Clean up temp directories
    temp_dirs = ["dataset/temp_real", "dataset/temp_ai", "dataset/temp_processed"]
    for temp_dir in temp_dirs:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
    
    return True

def main():
    """Main function to download and organize balanced dataset"""
    print("🚀 Downloading Balanced AI Face Detection Dataset")
    print("=" * 60)
    
    # Create dataset directory
    Path("dataset").mkdir(exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Download datasets
    real_success = download_real_faces()
    ai_success = download_ai_faces()
    
    if not (real_success and ai_success):
        print("❌ Failed to download required datasets")
        return False
    
    # Organize dataset
    if organize_dataset():
        print("\n🎉 Balanced dataset ready!")
        print("📊 Dataset structure:")
        
        base_dir = Path("dataset/balanced")
        for split in ['train', 'val']:
            for class_name in ['real', 'fake']:
                class_dir = base_dir / split / class_name
                if class_dir.exists():
                    count = len(list(class_dir.glob("*.jpg")))
                    print(f"   {split}/{class_name}: {count} images")
        
        print("\n🚀 Next step: python train_balanced.py")
        return True
    else:
        print("❌ Failed to organize dataset")
        return False

if __name__ == "__main__":
    main()