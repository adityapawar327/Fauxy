#!/usr/bin/env python3
"""
Organize existing balanced dataset for training
"""

import os
import shutil
import random
from pathlib import Path
from PIL import Image

def organize_balanced_dataset():
    """Organize the existing AI and real face dataset"""
    print("🚀 Organizing Existing Balanced Dataset")
    print("=" * 50)
    
    # Source directories
    source_dir = Path("dataset/AI-face-detection-Dataset")
    ai_source = source_dir / "AI"
    real_source = source_dir / "real"
    
    # Target directory
    target_dir = Path("dataset/balanced")
    
    # Check if source exists
    if not source_dir.exists():
        print("❌ Source dataset not found!")
        return False
    
    # Count images
    ai_images = list(ai_source.glob("*"))
    real_images = list(real_source.glob("*"))
    
    print(f"📊 Found {len(ai_images)} AI images and {len(real_images)} real images")
    
    # Create target structure
    for split in ['train', 'val']:
        for class_name in ['real', 'fake']:
            (target_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process AI images (fake class)
    print("🔄 Processing AI-generated images...")
    random.shuffle(ai_images)
    
    # Take up to 1000 AI images to balance with real images
    ai_images = ai_images[:1000]
    split_idx = int(len(ai_images) * 0.8)
    
    ai_train = ai_images[:split_idx]
    ai_val = ai_images[split_idx:]
    
    # Copy AI images
    for i, img_path in enumerate(ai_train):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                # Validate and resize image
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    new_name = f"fake_train_{i:04d}.jpg"
                    img.save(target_dir / "train" / "fake" / new_name, 'JPEG', quality=95)
            except Exception as e:
                print(f"⚠️ Skipped AI image {img_path}: {e}")
    
    for i, img_path in enumerate(ai_val):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    new_name = f"fake_val_{i:04d}.jpg"
                    img.save(target_dir / "val" / "fake" / new_name, 'JPEG', quality=95)
            except Exception as e:
                print(f"⚠️ Skipped AI image {img_path}: {e}")
    
    # Process real images
    print("🔄 Processing real images...")
    random.shuffle(real_images)
    
    # Take up to 1000 real images to balance with AI images
    real_images = real_images[:1000]
    split_idx = int(len(real_images) * 0.8)
    
    real_train = real_images[:split_idx]
    real_val = real_images[split_idx:]
    
    # Copy real images
    for i, img_path in enumerate(real_train):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    new_name = f"real_train_{i:04d}.jpg"
                    img.save(target_dir / "train" / "real" / new_name, 'JPEG', quality=95)
            except Exception as e:
                print(f"⚠️ Skipped real image {img_path}: {e}")
    
    for i, img_path in enumerate(real_val):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                with Image.open(img_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img = img.resize((224, 224), Image.Resampling.LANCZOS)
                    
                    new_name = f"real_val_{i:04d}.jpg"
                    img.save(target_dir / "val" / "real" / new_name, 'JPEG', quality=95)
            except Exception as e:
                print(f"⚠️ Skipped real image {img_path}: {e}")
    
    # Report final counts
    print("\n📊 Final dataset structure:")
    for split in ['train', 'val']:
        for class_name in ['real', 'fake']:
            class_dir = target_dir / split / class_name
            count = len(list(class_dir.glob("*.jpg")))
            print(f"   {split}/{class_name}: {count} images")
    
    print("\n🎉 Balanced dataset organized successfully!")
    print("🚀 Next step: python train_balanced.py")
    return True

if __name__ == "__main__":
    organize_balanced_dataset()