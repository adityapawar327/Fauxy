# 🤖 AI Face Detector

A modern web application that uses machine learning to detect AI-generated faces in images with high accuracy and real-time analysis.

![AI Face Detector](https://img.shields.io/badge/AI-Face%20Detector-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18+-blue?style=for-the-badge&logo=react)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)

## ✨ Features

- 🚀 **Real-time Detection**: Upload images and get instant AI detection results
- 🎯 **High Accuracy**: Uses a balanced CNN model trained on real and AI-generated faces
- 💻 **Modern UI**: Clean, responsive interface built with React and Tailwind CSS
- ⚡ **Fast API**: Built with FastAPI for high-performance backend processing
- 📊 **Detailed Analysis**: Provides confidence scores, artifact analysis, and texture evaluation
- 🔍 **Multiple Formats**: Supports JPG, PNG, WebP image formats
- 📱 **Responsive Design**: Works seamlessly on desktop and mobile devices

## 🛠️ Tech Stack

### Frontend

- **React 18** - Modern UI library
- **Vite** - Fast build tool and dev server
- **Tailwind CSS** - Utility-first CSS framework
- **JavaScript ES6+** - Modern JavaScript features

### Backend

- **FastAPI** - High-performance Python web framework
- **TensorFlow 2.x** - Machine learning framework
- **OpenCV** - Computer vision library
- **Pillow** - Image processing library
- **scikit-learn** - Machine learning utilities

### ML Model

- **Custom CNN Architecture** - Balanced training approach
- **Balanced Dataset** - Equal real and AI-generated face samples
- **Data Augmentation** - Improved model generalization

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn package manager

### 1. Clone & Setup

```bash
git clone <your-repo-url>
cd ai-face-detector

# Install Python dependencies
pip install -r backend/requirements.txt

# Install Node.js dependencies
cd frontend && npm install && cd ..
```

### 2. Kaggle API Setup

```bash
# Download kaggle.json from your Kaggle account settings
# Place it in the project root directory
# The file should contain your Kaggle API credentials
```

### 3. Dataset Preparation

```bash
# Option A: Organize existing dataset (if you have the data)
python scripts/organize_existing_dataset.py

# Option B: Download fresh balanced dataset
python scripts/download_balanced_dataset.py
```

### 4. Model Training

```bash
# Train the balanced model (recommended)
python scripts/train_balanced.py

# This will create a model trained on both real and AI faces
# Training takes 10-20 minutes depending on your hardware
```

### 5. Launch Application

```bash
# Option 1: One-command launcher (recommended)
python run.py

# Option 2: Manual startup
# Terminal 1 - Start backend
cd backend && python main.py

# Terminal 2 - Start frontend
cd frontend && npm run dev
```

### 6. Access the Application

- 🌐 **Frontend**: http://localhost:3000
- 📚 **API Documentation**: http://localhost:8000/docs
- ❤️ **Health Check**: http://localhost:8000/api/health

## 📁 Project Structure

```
ai-face-detector/
├── 🎨 frontend/                    # React Frontend Application
│   ├── src/                       # React components and logic
│   │   ├── components/           # Reusable UI components
│   │   ├── App.jsx              # Main application component
│   │   └── main.jsx             # Application entry point
│   ├── index.html               # Main HTML template
│   ├── package.json             # Frontend dependencies
│   ├── vite.config.js           # Vite build configuration
│   ├── tailwind.config.js       # Tailwind CSS configuration
│   └── postcss.config.js        # PostCSS configuration
│
├── 🔧 backend/                     # FastAPI Backend Server
│   ├── model/                    # ML Model and Detection Logic
│   │   ├── __init__.py          # Package initialization
│   │   ├── detector.py          # AI detection implementation
│   │   ├── ai_face_detector.h5  # Trained model file (generated)
│   │   └── *.png, *.txt         # Training artifacts (generated)
│   ├── main.py                  # FastAPI server and API endpoints
│   └── requirements.txt         # Python dependencies
│
├── 📜 scripts/                     # Utility and Setup Scripts
│   ├── train_balanced.py         # Train model with balanced dataset
│   ├── organize_existing_dataset.py  # Organize your existing data
│   ├── download_balanced_dataset.py  # Download fresh dataset
│   ├── test_setup.py             # Verify installation and setup
│   ├── setup_simple.py           # Simple setup automation
│   ├── train.py                  # Legacy training script
│   └── download_dataset.py       # Legacy download script
│
├── 📊 dataset/                     # Training Data
│   └── balanced/                 # Organized balanced dataset
│       ├── train/               # Training images
│       │   ├── real/           # Real face images
│       │   └── fake/           # AI-generated face images
│       └── val/                 # Validation images
│           ├── real/           # Real face validation
│           └── fake/           # AI-generated validation
│
├── 🚀 run.py                      # Main application launcher
├── 🔑 kaggle.json                 # Kaggle API credentials
├── 📖 README.md                   # This documentation
├── 🚫 .gitignore                  # Git ignore rules
└── ⚙️  .vscode/                   # VS Code settings (optional)
```

## 🔌 API Endpoints

| Method | Endpoint       | Description                             | Response                                    |
| ------ | -------------- | --------------------------------------- | ------------------------------------------- |
| `POST` | `/api/analyze` | Analyze uploaded image for AI detection | JSON with prediction, confidence, artifacts |
| `GET`  | `/api/health`  | Health check and model status           | JSON with status and model info             |
| `GET`  | `/api/stats`   | Get detector statistics and metrics     | JSON with performance stats                 |

### Example API Usage

```bash
# Health check
curl http://localhost:8000/api/health

# Analyze image
curl -X POST -F "file=@image.jpg" http://localhost:8000/api/analyze
```

## 📊 Model Performance

| Metric                | Value          | Description                            |
| --------------------- | -------------- | -------------------------------------- |
| **Architecture**      | Custom CNN     | Balanced training with real + AI faces |
| **Dataset Size**      | 2,000 images   | 1,000 real + 1,000 AI-generated faces  |
| **Training Split**    | 80/20          | 1,600 training, 400 validation images  |
| **Processing Time**   | ~1-2 seconds   | Per image analysis on CPU              |
| **Supported Formats** | JPG, PNG, WebP | Common image formats                   |
| **Max File Size**     | 10MB           | Upload limit for optimal performance   |
| **Input Resolution**  | 224x224 pixels | Automatically resized                  |

### Model Architecture

- **Input Layer**: 224x224x3 RGB images
- **Convolutional Layers**: 4 blocks with BatchNorm and Dropout
- **Pooling**: MaxPooling2D for feature reduction
- **Dense Layers**: 512 → 256 → 1 neurons
- **Activation**: ReLU (hidden), Sigmoid (output)
- **Optimizer**: Adam with learning rate scheduling

## 📜 Available Scripts

### 🎯 Main Commands

```bash
# 🚀 Launch entire application
python run.py

# 🧪 Test your setup
python scripts/test_setup.py

# 🏋️ Train the model
python scripts/train_balanced.py
```

### 🔧 Development Commands

```bash
# Backend only (API server)
cd backend && python main.py

# Frontend only (React dev server)
cd frontend && npm run dev

# Frontend build for production
cd frontend && npm run build
```

### 📊 Dataset Management

```bash
# Organize existing dataset
python scripts/organize_existing_dataset.py

# Download fresh balanced dataset
python scripts/download_balanced_dataset.py

# Download original dataset (legacy)
python scripts/download_dataset.py
```

### 🛠️ Utility Scripts

```bash
# Simple setup automation
python scripts/setup_simple.py

# Legacy training (unbalanced)
python scripts/train.py
```

## 🔬 Development Guide

### Setting Up Development Environment

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r backend/requirements.txt
cd frontend && npm install && cd ..

# 3. Verify setup
python scripts/test_setup.py
```

### Training Your Own Model

```bash
# 1. Prepare balanced dataset
python scripts/organize_existing_dataset.py

# 2. Train with custom parameters
python scripts/train_balanced.py

# 3. Monitor training progress
# Check backend/model/ for training artifacts:
# - balanced_training_history.png
# - balanced_confusion_matrix.png
# - balanced_classification_report.txt
```

### Adding New Features

1. **Backend**: Add endpoints in `backend/main.py`
2. **Frontend**: Add components in `frontend/src/components/`
3. **ML Model**: Modify `backend/model/detector.py`
4. **Scripts**: Add utilities in `scripts/`

## 🐛 Troubleshooting

### Common Issues & Solutions

| Issue                     | Symptoms                               | Solution                                                 |
| ------------------------- | -------------------------------------- | -------------------------------------------------------- |
| **Module not found**      | `ImportError` or `ModuleNotFoundError` | `pip install -r backend/requirements.txt`                |
| **Frontend won't start**  | `npm` errors or blank page             | `cd frontend && npm install`                             |
| **Port already in use**   | `Address already in use` error         | Change ports in config or kill existing processes        |
| **Model shows 99% AI**    | All images detected as AI              | Train balanced model: `python scripts/train_balanced.py` |
| **API connection failed** | Frontend can't reach backend           | Start backend first: `cd backend && python main.py`      |
| **Kaggle API errors**     | Dataset download fails                 | Check `kaggle.json` credentials                          |
| **Out of memory**         | Training crashes                       | Reduce batch size in training script                     |
| **Slow predictions**      | Long processing times                  | Use GPU or reduce image size                             |

### Debug Commands

```bash
# Check Python environment
python --version
pip list

# Check Node environment
node --version
npm --version

# Test API manually
curl http://localhost:8000/api/health

# Check ports
netstat -an | grep :8000
netstat -an | grep :3000
```

### Performance Optimization

- **GPU Training**: Install `tensorflow-gpu` for faster training
- **Model Size**: Use model quantization for smaller file size
- **Caching**: Enable browser caching for faster frontend loading
- **Batch Processing**: Process multiple images simultaneously

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Workflow

1. **Fork** the repository
2. **Clone** your fork: `git clone <your-fork-url>`
3. **Create** a feature branch: `git checkout -b feature/amazing-feature`
4. **Make** your changes
5. **Test** your changes: `python scripts/test_setup.py`
6. **Commit** your changes: `git commit -m 'Add amazing feature'`
7. **Push** to your branch: `git push origin feature/amazing-feature`
8. **Submit** a pull request

### Code Style

- **Python**: Follow PEP 8 guidelines
- **JavaScript**: Use ES6+ features and consistent formatting
- **Comments**: Document complex logic and API endpoints
- **Testing**: Add tests for new features

### Areas for Contribution

- 🎨 UI/UX improvements
- 🚀 Performance optimizations
- 🧪 Additional model architectures
- 📊 Better visualization and analytics
- 🔧 DevOps and deployment scripts
- 📖 Documentation improvements

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Dataset**: [Kaggle AI-Generated Face Detection Dataset](https://www.kaggle.com/datasets/shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset)
- **TensorFlow Team**: For the excellent ML framework
- **FastAPI Team**: For the high-performance web framework
- **React Team**: For the modern UI library
- **Tailwind CSS**: For the utility-first CSS framework
- **Vite Team**: For the fast build tool