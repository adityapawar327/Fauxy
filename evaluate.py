#!/usr/bin/env python3
"""
Evaluation script for AI Face Detector
Test the trained model on sample images
"""

import asyncio
import sys
from pathlib import Path
from PIL import Image
import argparse

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

from model.detector import AIFaceDetector

async def evaluate_image(image_path: str):
    """Evaluate a single image"""
    detector = AIFaceDetector()
    
    print("🔄 Loading model...")
    await detector.load_model()
    
    if not detector.model_loaded:
        print("❌ Failed to load model")
        return
    
    try:
        # Load and analyze image
        image = Image.open(image_path)
        print(f"📸 Analyzing: {image_path}")
        print(f"📏 Image size: {image.size}")
        
        result = await detector.predict(image)
        
        # Display results
        print("\n🎯 Analysis Results:")
        print("=" * 40)
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"Raw Score: {result.get('raw_prediction', 'N/A'):.3f}")
        
        print("\n🔍 Detailed Analysis:")
        print(f"Artifacts: {result['artifacts']}")
        print(f"Consistency: {result['consistency']}")
        print(f"Texture: {result['texture']}")
        print(f"Symmetry: {result['symmetry']}")
        
        # Interpretation
        print(f"\n💡 Interpretation:")
        if result['prediction'] == 'AI Generated':
            print("⚠️  This image appears to be artificially generated")
            if result['confidence'] > 0.8:
                print("🔴 High confidence - likely AI generated")
            else:
                print("🟡 Medium confidence - possibly AI generated")
        else:
            print("✅ This image appears to be authentic/real")
            if result['confidence'] > 0.8:
                print("🟢 High confidence - likely real")
            else:
                print("🟡 Medium confidence - possibly real")
                
    except Exception as e:
        print(f"❌ Error analyzing image: {e}")

async def evaluate_directory(directory_path: str):
    """Evaluate all images in a directory"""
    detector = AIFaceDetector()
    
    print("🔄 Loading model...")
    await detector.load_model()
    
    if not detector.model_loaded:
        print("❌ Failed to load model")
        return
    
    directory = Path(directory_path)
    if not directory.exists():
        print(f"❌ Directory not found: {directory_path}")
        return
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(directory.glob(f"*{ext}"))
        image_files.extend(directory.glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No image files found in {directory_path}")
        return
    
    print(f"📁 Found {len(image_files)} images")
    
    results = []
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"\n[{i}/{len(image_files)}] {image_path.name}")
            
            image = Image.open(image_path)
            result = await detector.predict(image)
            
            results.append({
                'file': image_path.name,
                'prediction': result['prediction'],
                'confidence': result['confidence']
            })
            
            print(f"  → {result['prediction']} ({result['confidence']:.1%})")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    # Summary
    print(f"\n📊 Summary of {len(results)} analyzed images:")
    print("=" * 50)
    
    ai_count = sum(1 for r in results if r['prediction'] == 'AI Generated')
    real_count = len(results) - ai_count
    
    print(f"🤖 AI Generated: {ai_count} ({ai_count/len(results)*100:.1f}%)")
    print(f"👤 Real/Authentic: {real_count} ({real_count/len(results)*100:.1f}%)")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    print(f"📈 Average Confidence: {avg_confidence:.1%}")

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate AI Face Detector')
    parser.add_argument('path', help='Path to image file or directory')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all images in directory')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if not path.exists():
        print(f"❌ Path not found: {args.path}")
        return
    
    print("🔍 AI Face Detector - Evaluation")
    print("=" * 40)
    
    if path.is_file() and not args.batch:
        # Single image
        asyncio.run(evaluate_image(str(path)))
    elif path.is_dir() or args.batch:
        # Directory
        if path.is_file():
            path = path.parent
        asyncio.run(evaluate_directory(str(path)))
    else:
        print("❌ Invalid path or use --batch for directory processing")

if __name__ == "__main__":
    main()