#!/usr/bin/env python3
"""
Simple setup script for AI Face Detector
Downloads dataset and trains the model
"""

import os
import sys
from pathlib import Path

def main():
    print("🚀 AI Face Detector - Simple Setup")
    print("=" * 50)
    
    # Step 1: Download dataset
    print("📥 Step 1: Downloading dataset...")
    try:
        import download_dataset
        success = download_dataset.download_dataset()
        if not success:
            print("❌ Dataset download failed!")
            return
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("💡 Make sure you have kaggle.json configured")
        return
    
    # Step 2: Train model
    print("\n🎯 Step 2: Training model...")
    try:
        from scripts.train_model import AIFaceTrainer
        trainer = AIFaceTrainer(data_dir="dataset/organized")
        model, history = trainer.train()
        
        if model is not None:
            print("\n🎉 Setup completed successfully!")
            print("\n📋 What's ready:")
            print("✅ Dataset downloaded and organized")
            print("✅ Model trained and saved")
            print("✅ Web interface ready")
            print("\n🚀 Start the app:")
            print("   python main.py")
            print("\n🌐 Then open: http://localhost:8000")
        else:
            print("❌ Training failed!")
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()