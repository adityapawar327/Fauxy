#!/usr/bin/env python3
"""
Simple script to download the Kaggle AI face detection dataset
"""

import kaggle
import os
import shutil
from pathlib import Path

def download_dataset():
    """Download and organize the AI face detection dataset"""
    
    dataset_name = "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset"
    download_path = "./dataset"
    
    print("🔄 Downloading AI Face Detection Dataset...")
    print(f"📦 Dataset: {dataset_name}")
    
    try:
        # Create dataset directory
        os.makedirs(download_path, exist_ok=True)
        
        # Download dataset
        print("📥 Starting download (this may take a few minutes)...")
        kaggle.api.dataset_download_files(
            dataset_name, 
            path=download_path, 
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        
        # Show what we got
        dataset_path = Path(download_path)
        all_files = list(dataset_path.rglob("*"))
        image_files = [f for f in all_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        print(f"📊 Found {len(image_files)} images")
        
        # Try to organize into real/fake folders
        organize_dataset(dataset_path, image_files)
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Make sure your kaggle.json is in ~/.kaggle/ directory")
        return False

def organize_dataset(dataset_path, image_files):
    """Organize images into real/fake folders"""
    print("📁 Organizing dataset...")
    
    # Create organized structure
    train_dir = dataset_path / "organized" / "train"
    val_dir = dataset_path / "organized" / "val"
    
    for split in [train_dir, val_dir]:
        (split / "real").mkdir(parents=True, exist_ok=True)
        (split / "fake").mkdir(parents=True, exist_ok=True)
    
    real_count = 0
    fake_count = 0
    
    for i, img_file in enumerate(image_files):
        # Determine if real or fake based on path/filename
        path_str = str(img_file).lower()
        is_fake = any(keyword in path_str for keyword in [
            'fake', 'ai', 'generated', 'synthetic', 'artificial', 'gan'
        ])
        
        # 80/20 train/val split
        is_train = i % 5 != 0  # 80% train, 20% val
        
        if is_fake:
            dest_dir = train_dir / "fake" if is_train else val_dir / "fake"
            fake_count += 1
        else:
            dest_dir = train_dir / "real" if is_train else val_dir / "real"
            real_count += 1
        
        # Copy file with simple naming
        dest_file = dest_dir / f"{len(list(dest_dir.glob('*')))}.jpg"
        try:
            shutil.copy2(img_file, dest_file)
        except Exception as e:
            print(f"⚠️  Skipped {img_file}: {e}")
    
    print(f"✅ Organized: {real_count} real, {fake_count} fake images")
    print(f"📈 Split: 80% train, 20% validation")

if __name__ == "__main__":
    success = download_dataset()
    if success:
        print("\n🎉 Dataset ready!")
        print("📝 Next: Run 'python train.py' to train the model")
    else:
        print("\n❌ Download failed. Check your Kaggle setup.")