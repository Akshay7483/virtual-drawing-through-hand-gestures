# ✋ Virtual Drawing Through Hand Gestures

> AI-Powered virtual drawing running **100% in your browser** — no downloads, no backend needed!

![JavaScript](https://img.shields.io/badge/JavaScript-ES2022-yellow?logo=javascript)
![MediaPipe](https://img.shields.io/badge/MediaPipe-JS-orange)
![GitHub Pages](https://img.shields.io/badge/Deployed-GitHub%20Pages-blue?logo=github)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌐 Live Demo

**👉 [Try it now!](https://Akshay7483.github.io/virtual-drawing-through-hand-gestures/)**

Just open the link, allow camera access, and start drawing with your hands!

---

## 🎯 What is This?

A real-time virtual drawing application that lets you **draw in the air using hand gestures** detected through your webcam. Powered by **MediaPipe AI** running entirely in the browser using WebAssembly and WebGL.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **Smart Tracking** | EMA smoothing for perfectly stable lines |
| ✨ **Shape Recognition** | Auto-correct rough strokes into perfect circles, rectangles, lines |
| ↩️ **Gesture Undo/Redo** | Fist = Undo, Open palm = Redo |
| 🎨 **6-Color Palette** | Blue, Green, Red, Yellow, Cyan, Purple |
| 📊 **Live Statistics** | Real-time stroke count, time, and shape stats |
| 🖐️ **Multi-Hand Support** | Track up to 2 hands simultaneously |
| 🧹 **Eraser & Clear** | Erase strokes or clear the entire canvas |
| 💾 **Save Drawings** | Download your artwork as PNG |
| 🌐 **100% Browser** | No installation, no backend, works on any device |

---

## 🖐️ Gesture Guide

| Gesture | Action |
|---|---|
| ☝️ **Index finger only** | Draw on canvas |
| ✌️ **Peace sign** (index + middle) | Select toolbar tools |
| ✊ **Fist** (all fingers down) | Undo last stroke |
| 🖐️ **Open palm** (all fingers up) | Redo |

---

## 🚀 How to Use

1. Open the [live demo](https://Akshay7483.github.io/virtual-drawing-through-hand-gestures/)
2. Click **"Start Camera"**
3. Allow camera access when prompted
4. Raise your **index finger** to start drawing!
5. Use the toolbar on the left to change colors, erase, or toggle shape recognition

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| **MediaPipe Tasks Vision (JS)** | AI hand landmark detection in browser |
| **HTML5 Canvas** | Drawing and rendering |
| **JavaScript (ES Modules)** | Core application logic |
| **WebAssembly + WebGL** | GPU-accelerated ML inference |
| **GitHub Pages** | Free hosting and deployment |

---

## 📁 Project Structure

```
virtual-drawing-through-hand-gestures/
├── src/
│   ├── backend/
│   │   ├── draw_hand_gestures.py   # Python version (local use)
│   │   └── ai_features.py          # AI features module
│   └── frontend/
│       ├── index.html              # Main web app
│       ├── style.css               # Dark theme styling
│       └── script.js               # MediaPipe JS + drawing engine
├── requirements.txt                # Python dependencies (for local version)
├── START_DRAWING.bat               # Windows launcher (for local version)
└── README.md
```

---

## 🤖 AI Features

### Shape Auto-Correction
Draw a rough circle or rectangle — the AI recognizes and auto-corrects it into a perfect geometric shape. Toggle with the **Shapes** button.

### EMA Smoothing
Exponential Moving Average filter (α=0.4) smooths hand landmark coordinates for jitter-free drawing.

### Gesture Recognition
5-finger state detection using landmark tip/pip comparison to recognize Draw, Select, Undo, Redo, and Idle gestures.

---

## 💻 Run Locally (Python Version)

```bash
git clone https://github.com/Akshay7483/virtual-drawing-through-hand-gestures.git
cd virtual-drawing-through-hand-gestures
pip install -r requirements.txt
cd src/backend
python draw_hand_gestures.py
```

---

## 👨‍💻 Author

**Akshay Penumarthi**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
