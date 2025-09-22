#!/usr/bin/env python3
"""
Setup script for AI Face Detector
Downloads and prepares the Kaggle dataset
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

def install_kaggle():
    """Install Kaggle API if not present"""
    try:
        # Check if kaggle package is installed without importing it
        subprocess.check_call([sys.executable, "-c", "import kaggle"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✓ Kaggle API already installed")
    except subprocess.CalledProcessError:
        print("Installing Kaggle API...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✓ Kaggle API installed")

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    credentials_file = kaggle_dir / "kaggle.json"
    
    if not credentials_file.exists():
        print("\n⚠️  Kaggle credentials not found!")
        print("Please follow these steps:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Click 'Create New API Token'")
        print("3. Save the kaggle.json file to:", credentials_file)
        print("4. Run this script again")
        return False
    
    # Set proper permissions
    credentials_file.chmod(0o600)
    print("✓ Kaggle credentials configured")
    return True

def download_dataset():
    """Download the AI faces dataset from Kaggle"""
    dataset_name = "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print(f"📥 Downloading dataset: {dataset_name}")
    print("This may take several minutes depending on your internet connection...")
    
    try:
        # Import kaggle only when we need it, after credentials are set up
        import kaggle
        
        # Download the dataset
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )
        
        # Check if files were extracted properly
        extracted_files = list(data_dir.rglob("*"))
        print(f"✓ Dataset downloaded successfully")
        print(f"✓ Extracted {len(extracted_files)} files")
        
        # Print dataset structure
        print("\n📁 Dataset structure:")
        for item in sorted(data_dir.iterdir()):
            if item.is_dir():
                file_count = len(list(item.glob("*")))
                print(f"  {item.name}/  ({file_count} files)")
            else:
                print(f"  {item.name}")
        
        return True
        
    except OSError as e:
        if "kaggle.json" in str(e):
            print("❌ Kaggle credentials not found!")
            print("\n🔑 Please set up Kaggle API credentials:")
            print("1. Go to https://www.kaggle.com/account")
            print("2. Click 'Create New API Token'")
            print("3. Save kaggle.json to: C:\\Users\\Adity\\.kaggle\\")
            print("4. Run this script again")
            return False
        else:
            print(f"❌ Failed to download dataset: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to download dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure you have Kaggle API credentials set up")
        print("2. Check your internet connection")
        print("3. Verify the dataset name is correct")
        print("4. Try downloading manually from:")
        print(f"   https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
        return False

def setup_model_directory():
    """Create model directory structure"""
    dirs = [
        "model/weights",
        "model/checkpoints",
        "model/logs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✓ Model directories created")

def main():
    """Main setup function"""
    print("🚀 Setting up AI Face Detector...")
    
    # Install dependencies
    install_kaggle()
    
    # Setup Kaggle credentials
    if not setup_kaggle_credentials():
        return
    
    # Download dataset
    if download_dataset():
        print("✓ Dataset ready for training")
    
    # Setup model directories
    setup_model_directory()
    
    print("\n🎉 Setup complete!")
    print("\nNext steps:")
    print("1. Install frontend dependencies: npm install")
    print("2. Install Python dependencies: pip install -r requirements.txt")
    print("3. Train your model using the downloaded dataset")
    print("4. Start the application: npm run dev & python main.py")

if __name__ == "__main__":
    main()