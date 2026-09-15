# ✋ Virtual Drawing Through Hand Gestures

> AI-Powered virtual drawing system using hand gestures detected through your webcam.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-5.0-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🎯 What is This?

A real-time virtual drawing application that lets you **draw in the air using hand gestures** detected through your webcam. Powered by **MediaPipe AI** for hand tracking and **OpenCV** for rendering.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Smart Tracking** | EMA smoothing for perfectly stable lines |
| ✨ **Shape Recognition** | Auto-correct rough strokes into perfect circles, rectangles, triangles |
| ↩️ **Gesture Undo/Redo** | Fist = Undo, Open palm = Redo |
| 🎨 **6-Color Palette** | Blue, Green, Red, Yellow, Cyan, Purple |
| 📊 **Drawing Statistics** | Real-time stats overlay (strokes, time, shapes) |
| 🖐️ **Multi-Hand Support** | Track up to 2 hands simultaneously |
| 🧹 **Eraser & Clear** | Erase strokes or clear the entire canvas |
| 💾 **Save Drawings** | Save your artwork as PNG files |

---

## 🖐️ Gesture Guide

| Gesture | Action |
|---|---|
| ☝️ **Index finger only** | Draw on canvas |
| ✌️ **Peace sign** (index + middle) | Select toolbar tools |
| ✊ **Fist** (all fingers down) | Undo last stroke |
| 🖐️ **Open palm** (all fingers up) | Redo |
| Press **Q** | Quit the application |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Akshay7483/virtual-drawing-through-hand-gestures.git
cd virtual-drawing-through-hand-gestures
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the AI model
```bash
cd src/backend
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'hand_landmarker.task'); print('Model downloaded!')"
```

### 4. Run the application
```bash
python draw_hand_gestures.py
```

Or simply **double-click** `START_DRAWING.bat` on Windows!

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **Python** | Core language |
| **MediaPipe 1.0** | AI hand landmark detection |
| **OpenCV** | Computer vision & rendering |
| **NumPy** | Array operations |
| **SciPy** | Gaussian smoothing for strokes |

---

## 📁 Project Structure

```
virtual-drawing-through-hand-gestures/
├── src/
│   ├── backend/
│   │   ├── draw_hand_gestures.py   # Main drawing engine
│   │   ├── ai_features.py          # AI: shape recognition, smoothing, stats
│   │   └── hand_landmarker.task    # MediaPipe model (download required)
│   └── frontend/
│       ├── index.html              # Project showcase page
│       ├── style.css               # Dark theme styling
│       └── script.js               # Particle animations
├── requirements.txt
├── START_DRAWING.bat               # Windows launcher
└── README.md
```

---

## 🤖 AI Features

### Shape Auto-Correction
Draw a rough circle, rectangle, or triangle — the AI will recognize and auto-correct it into a perfect geometric shape. Toggle with the **SHAPES** button in the toolbar.

### EMA Smoothing
Exponential Moving Average filter (α=0.4) smooths raw hand landmark coordinates, eliminating jitter for precise drawing.

### Jitter Filtering
A 3-pixel minimum movement threshold filters out micro-jitters from hand tremors.

---

## 📸 Toolbar

The on-screen toolbar includes:
- **Row 1**: BLUE | GREEN | RED | YELLOW | CYAN | PURPLE
- **Row 2**: ERASE | CLEAR | UNDO | REDO | SHAPES | SAVE | EXIT

---

## 🌐 Live Demo Page

Visit the project showcase: [GitHub Pages](https://Akshay7483.github.io/virtual-drawing-through-hand-gestures/)

---

## 📌 Future Improvements

- [ ] Browser-based MediaPipe JS version
- [ ] Save drawings to cloud
- [ ] Multi-hand collaborative drawing
- [ ] Voice commands integration

---

## 👨‍💻 Author

**Akshay Penumarthi**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
