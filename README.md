# ISRO IROUC 2026 - QR Drone Detector

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV Version](https://img.shields.io/badge/opencv-4.5%2B-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/license-MIT-red.svg)](LICENSE)

A real-time QR code detection system with seed image matching and orientation feedback for drone tracking applications, developed for ISRO IROUC 2026.

## 🚀 Features

- **Real-time QR Detection**: Aggressive tracking with 60+ FPS performance
- **Seed Image Matching**: Load and match QR codes against a reference image
- **Orientation Feedback**: Real-time guidance for optimal QR code alignment
- **Multiple Detection Strategies**: 6 different processing techniques for robust detection
- **Kalman Filter Tracking**: Smooth tracking even when QR is temporarily lost
- **Low Latency**: Optimized threading for minimal delay
- **Interactive Controls**: Easy keyboard shortcuts for loading seeds and resetting

## 📋 Requirements

- Python 3.7+
- OpenCV 4.5+
- NumPy
- Tkinter (usually comes with Python)
- PyYAML (optional, for configuration)

## 🔧 Installation

### Quick Install

```bash
git clone https://github.com/yourusername/isro-qr-drone-detector.git
cd isro-qr-drone-detector
pip install -r requirements.txt
python main.py
