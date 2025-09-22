#!/usr/bin/env python3
"""
Simple Kaggle setup guide for AI Face Detector
"""

import os
import sys
from pathlib import Path
import json

def create_kaggle_dir():
    """Create .kaggle directory if it doesn't exist"""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    return kaggle_dir

def check_kaggle_credentials():
    """Check if Kaggle credentials exist"""
    kaggle_dir = Path.home() / ".kaggle"
    credentials_file = kaggle_dir / "kaggle.json"
    return credentials_file.exists()

def setup_credentials_interactive():
    """Interactive setup for Kaggle credentials"""
    print("🔑 Kaggle API Setup")
    print("=" * 30)
    
    if check_kaggle_credentials():
        print("✅ Kaggle credentials already exist!")
        return True
    
    print("📋 Follow these steps to get your Kaggle API credentials:")
    print()
    print("1. Go to: https://www.kaggle.com/account")
    print("2. Scroll down to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This will download 'kaggle.json' file")
    print()
    
    input("Press Enter when you have downloaded kaggle.json...")
    
    # Ask user for the downloaded file location
    print("\n📁 Where did you save kaggle.json?")
    print("Common locations:")
    print("1. Downloads folder")
    print("2. Desktop")
    print("3. Custom location")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        source_path = Path.home() / "Downloads" / "kaggle.json"
    elif choice == "2":
        source_path = Path.home() / "Desktop" / "kaggle.json"
    else:
        custom_path = input("Enter full path to kaggle.json: ").strip()
        source_path = Path(custom_path)
    
    if not source_path.exists():
        print(f"❌ File not found: {source_path}")
        print("Please make sure you downloaded kaggle.json and try again.")
        return False
    
    # Copy to correct location
    kaggle_dir = create_kaggle_dir()
    dest_path = kaggle_dir / "kaggle.json"
    
    try:
        # Read and copy the file
        with open(source_path, 'r') as src:
            credentials = json.load(src)
        
        with open(dest_path, 'w') as dst:
            json.dump(credentials, dst)
        
        # Set proper permissions (important for security)
        if os.name != 'nt':  # Not Windows
            os.chmod(dest_path, 0o600)
        
        print(f"✅ Credentials copied to: {dest_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to copy credentials: {e}")
        return False

def test_kaggle_connection():
    """Test if Kaggle API is working"""
    try:
        import kaggle
        
        print("\n🧪 Testing Kaggle connection...")
        
        # Try to list datasets (this will test authentication)
        datasets = kaggle.api.dataset_list(search="test", page_size=1)
        print("✅ Kaggle API connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Kaggle API test failed: {e}")
        return False

def download_ai_face_dataset():
    """Download the AI face detection dataset"""
    dataset_name = "shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset"
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    print(f"\n📥 Downloading dataset: {dataset_name}")
    print("⚠️  This is a large dataset (~several GB). It may take 10-30 minutes.")
    
    proceed = input("Do you want to proceed? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Download cancelled.")
        return False
    
    try:
        import kaggle
        
        print("🔄 Starting download...")
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(data_dir),
            unzip=True
        )
        
        print("✅ Dataset downloaded successfully!")
        
        # Show dataset structure
        print("\n📁 Dataset structure:")
        for item in sorted(data_dir.iterdir()):
            if item.is_dir():
                file_count = len(list(item.glob("*")))
                print(f"  {item.name}/  ({file_count} files)")
        
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 AI Face Detector - Kaggle Setup")
    print("=" * 50)
    
    # Step 1: Setup credentials
    if not setup_credentials_interactive():
        print("\n❌ Setup failed. Please try again.")
        return
    
    # Step 2: Test connection
    if not test_kaggle_connection():
        print("\n❌ Connection test failed. Please check your credentials.")
        return
    
    # Step 3: Download dataset
    if download_ai_face_dataset():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Train the model: python train.py")
        print("2. Start the application: python main.py")
    else:
        print("\n⚠️  Dataset download failed, but credentials are set up.")
        print("You can try downloading later with: python scripts/setup.py")

if __name__ == "__main__":
    main()