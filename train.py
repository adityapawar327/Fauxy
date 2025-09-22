#!/usr/bin/env python3
"""
Main training script for AI Face Detector
Run this to train the model with the Kaggle dataset
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from scripts.train_model import AIFaceTrainer

def main():
    """Main training function"""
    print("🚀 AI Face Detector Training")
    print("=" * 50)
    
    # Check if dataset exists
    data_dir = Path("data")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("❌ Dataset not found!")
        print("\n📥 Please download the dataset first:")
        print("1. Run: python scripts/setup.py")
        print("2. Or manually download from:")
        print("   https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
        print("3. Extract to the 'data/' folder")
        return
    
    # Initialize trainer
    trainer = AIFaceTrainer()
    
    # Create model directory
    trainer.model_dir.mkdir(exist_ok=True)
    
    print(f"📂 Data directory: {data_dir.absolute()}")
    print(f"📂 Model directory: {trainer.model_dir.absolute()}")
    
    # Start training
    try:
        model, history = trainer.train()
        
        if model is not None:
            print("\n🎉 Training completed successfully!")
            print("\n📊 Model Performance Summary:")
            print("- Check 'model/confusion_matrix.png' for detailed results")
            print("- Check 'model/training_history.png' for training curves")
            print("- Check 'model/roc_curve.png' for ROC analysis")
            print("\n🚀 Ready to use! Start the application with:")
            print("  python main.py")
        else:
            print("❌ Training failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()