#!/usr/bin/env python3
"""
Installation script for AI Face Detector
Handles dependency installation with compatibility checks
"""

import subprocess
import sys
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Try to install with specific versions first
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("⚠️ Specific versions failed, trying with latest compatible versions...")
        
        # Fallback to more flexible versions
        fallback_packages = [
            "fastapi>=0.100.0",
            "uvicorn>=0.20.0", 
            "python-multipart",
            "pillow>=10.0.0",
            "tensorflow>=2.16.0",
            "opencv-python>=4.8.0",
            "numpy>=1.24.0",
            "scikit-learn>=1.3.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "kaggle>=1.5.0",
            "aiofiles",
            "pydantic>=2.0.0"
        ]
        
        for package in fallback_packages:
            if not run_command(f"pip install {package}", f"Installing {package}"):
                print(f"⚠️ Failed to install {package}, continuing...")

def install_node_dependencies():
    """Install Node.js dependencies"""
    if not Path("package.json").exists():
        print("❌ package.json not found")
        return False
    
    # Check if npm is available
    try:
        subprocess.run(["npm", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm not found. Please install Node.js first:")
        print("   https://nodejs.org/")
        return False
    
    return run_command("npm install", "Installing Node.js dependencies")

def main():
    """Main installation function"""
    print("🚀 AI Face Detector - Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check platform
    print(f"💻 Platform: {platform.system()} {platform.machine()}")
    
    # Install Python dependencies
    install_dependencies()
    
    # Install Node.js dependencies
    install_node_dependencies()
    
    print("\n🎉 Installation completed!")
    print("\n📋 Next steps:")
    print("1. Setup Kaggle credentials: python scripts/setup.py")
    print("2. Train the model: python train.py")
    print("3. Start the application:")
    print("   - Frontend: npm run dev")
    print("   - Backend: python main.py")
    
    print("\n💡 Tips:")
    print("- Use a GPU for faster training")
    print("- Training will take 2-4 hours on GPU, 8-12 hours on CPU")
    print("- Check README.md for detailed instructions")

if __name__ == "__main__":
    main()