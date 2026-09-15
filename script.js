import { HandLandmarker, FilesetResolver } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

// Elements
const video = document.createElement('video');
video.setAttribute('playsinline', '');
video.style.display = 'none';

let videoCanvas, drawCanvas, uiCanvas;
let videoCtx, drawCtx, uiCtx;
let handLandmarker;
let isRunning = false;
let animationFrameId = null;

// Drawing state
let xp = 0, yp = 0;
let currentColor = '#3350FF';
let currentThickness = 5;
let isEraser = false;
let currentStroke = []; // Current drawing points for shape recognition
let shapeMode = false;
let gestureMode = 'IDLE';

// Smoothing
class SmoothLandmark {
  constructor(alpha = 0.4) {
    this.alpha = alpha;
    this.prev = null;
  }
  update(x, y) {
    if (!this.prev) { this.prev = { x, y }; return { x, y }; }
    const sx = this.alpha * x + (1 - this.alpha) * this.prev.x;
    const sy = this.alpha * y + (1 - this.alpha) * this.prev.y;
    this.prev = { x: sx, y: sy };
    return { x: Math.round(sx), y: Math.round(sy) };
  }
}
const smoother = new SmoothLandmark(0.4);

// Undo/Redo
let undoStack = [];
let redoStack = [];
const MAX_UNDO = 20;
let lastUndoTime = 0;
let lastRedoTime = 0;
const COOLDOWN = 1000;

// Stats
const stats = {
  strokeCount: 0,
  colorsUsed: new Set([currentColor]),
  startTime: null,
  shapesRecognized: 0
};

// Hand Connections
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],  // thumb
  [0,5],[5,6],[6,7],[7,8],  // index
  [5,9],[9,10],[10,11],[11,12],  // middle
  [9,13],[13,14],[14,15],[15,16],  // ring
  [13,17],[17,18],[18,19],[19,20],  // pinky
  [0,17]  // palm
];

// Initialize MediaPipe
async function initMediaPipe() {
  try {
    const vision = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
    );
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
        delegate: 'GPU'
      },
      numHands: 2,
      runningMode: 'VIDEO',
      minHandDetectionConfidence: 0.7,
      minHandPresenceConfidence: 0.7,
      minTrackingConfidence: 0.7
    });
    console.log("MediaPipe Initialized");
  } catch (err) {
    console.error("Error initializing MediaPipe:", err);
  }
}

// Particle Background (Simplified)
const particles = [];
let particleCanvas, particleCtx;
function initParticles() {
  particleCanvas = document.getElementById('particleCanvas');
  if (!particleCanvas) {
    particleCanvas = document.createElement('canvas');
    particleCanvas.id = 'particleCanvas';
    particleCanvas.style.position = 'fixed';
    particleCanvas.style.top = '0';
    particleCanvas.style.left = '0';
    particleCanvas.style.width = '100vw';
    particleCanvas.style.height = '100vh';
    particleCanvas.style.zIndex = '-1';
    particleCanvas.style.pointerEvents = 'none';
    document.body.appendChild(particleCanvas);
  }
  particleCtx = particleCanvas.getContext('2d');
  
  function resizeParticles() {
    particleCanvas.width = window.innerWidth;
    particleCanvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resizeParticles);
  resizeParticles();

  for (let i = 0; i < 50; i++) {
    particles.push({
      x: Math.random() * particleCanvas.width,
      y: Math.random() * particleCanvas.height,
      vx: (Math.random() - 0.5) * 2,
      vy: (Math.random() - 0.5) * 2,
      radius: Math.random() * 3 + 1
    });
  }
  
  function drawParticles() {
    if (isRunning) {
        particleCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
        requestAnimationFrame(drawParticles);
        return;
    }
    particleCtx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
    particleCtx.fillStyle = 'rgba(200, 200, 255, 0.5)';
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > particleCanvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > particleCanvas.height) p.vy *= -1;
      particleCtx.beginPath();
      particleCtx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      particleCtx.fill();
    });
    requestAnimationFrame(drawParticles);
  }
  drawParticles();
}

// Canvas Setup
function setupCanvases() {
  const container = document.getElementById('canvasContainer');
  if (!container) return;

  const width = 1280;
  const height = 720;
  
  // Set container style if not set in CSS
  container.style.position = 'relative';
  container.style.width = '100%';
  container.style.maxWidth = `${width}px`;
  container.style.aspectRatio = '16/9';
  container.style.margin = '0 auto';
  container.style.overflow = 'hidden';

  // Video Canvas
  videoCanvas = document.getElementById('videoCanvas') || document.createElement('canvas');
  videoCanvas.id = 'videoCanvas';
  videoCtx = videoCanvas.getContext('2d');
  
  // Draw Canvas
  drawCanvas = document.getElementById('drawCanvas') || document.createElement('canvas');
  drawCanvas.id = 'drawCanvas';
  drawCtx = drawCanvas.getContext('2d', { willReadFrequently: true });
  
  // UI Canvas
  uiCanvas = document.getElementById('uiCanvas') || document.createElement('canvas');
  uiCanvas.id = 'uiCanvas';
  uiCtx = uiCanvas.getContext('2d');
  
  [videoCanvas, drawCanvas, uiCanvas].forEach(c => {
    c.width = width;
    c.height = height;
    c.style.position = 'absolute';
    c.style.top = '0';
    c.style.left = '0';
    c.style.width = '100%';
    c.style.height = '100%';
    if (!c.parentElement) container.appendChild(c);
  });
}

// Finger/Gesture Logic
function countFingersUp(landmarks) {
  const fingers = [];
  // Thumb: tip (4) x vs ip (3) x (mirrored)
  fingers.push(landmarks[4].x > landmarks[3].x); 
  
  // Other fingers (tip y vs pip y)
  fingers.push(landmarks[8].y < landmarks[6].y);
  fingers.push(landmarks[12].y < landmarks[10].y);
  fingers.push(landmarks[16].y < landmarks[14].y);
  fingers.push(landmarks[20].y < landmarks[18].y);
  return fingers;
}

function getGesture(fingers) {
  const upCount = fingers.filter(f => f).length;
  if (fingers[1] && !fingers[2] && !fingers[3] && !fingers[4]) return 'DRAW';
  if (fingers[1] && fingers[2] && !fingers[3] && !fingers[4]) return 'SELECT';
  if (upCount === 0) return 'UNDO';
  if (upCount === 5) return 'REDO';
  return 'IDLE';
}

// Undo/Redo Functions
function saveState() {
  if (undoStack.length >= MAX_UNDO) undoStack.shift();
  undoStack.push(drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
  redoStack = []; // clear redo on new action
}

function performUndo() {
  if (undoStack.length > 0) {
    redoStack.push(drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
    const state = undoStack.pop();
    drawCtx.putImageData(state, 0, 0);
  } else {
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  }
}

function performRedo() {
  if (redoStack.length > 0) {
    undoStack.push(drawCtx.getImageData(0, 0, drawCanvas.width, drawCanvas.height));
    const state = redoStack.pop();
    drawCtx.putImageData(state, 0, 0);
  }
}

// Shape Recognition
function recognizeShape(points) {
  if (points.length < 10) return null;
  
  const minX = Math.min(...points.map(p => p.x));
  const maxX = Math.max(...points.map(p => p.x));
  const minY = Math.min(...points.map(p => p.y));
  const maxY = Math.max(...points.map(p => p.y));
  const w = maxX - minX, h = maxY - minY;
  
  const first = points[0], last = points[points.length - 1];
  const closeDist = Math.sqrt((first.x-last.x)**2 + (first.y-last.y)**2);
  const isClosed = closeDist < Math.max(w, h) * 0.3;
  
  if (!isClosed) return { type: 'line', start: first, end: last };
  
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const avgRadius = points.reduce((sum, p) => sum + Math.sqrt((p.x-cx)**2 + (p.y-cy)**2), 0) / points.length;
  const radiusVariance = points.reduce((sum, p) => {
    const r = Math.sqrt((p.x-cx)**2 + (p.y-cy)**2);
    return sum + (r - avgRadius) ** 2;
  }, 0) / points.length;
  const circularity = 1 - Math.sqrt(radiusVariance) / avgRadius;
  
  if (circularity > 0.85) return { type: 'circle', cx, cy, radius: avgRadius };
  
  const aspectRatio = w / h;
  if (aspectRatio > 0.7 && aspectRatio < 1.4) {
    return { type: 'rectangle', x: minX, y: minY, w, h };
  }
  return { type: 'rectangle', x: minX, y: minY, w, h };
}

function drawPerfectShape(ctx, shape, color, thickness) {
  ctx.strokeStyle = color;
  ctx.lineWidth = thickness;
  ctx.lineCap = 'round';
  if (shape.type === 'circle') {
    ctx.beginPath();
    ctx.arc(shape.cx, shape.cy, shape.radius, 0, Math.PI * 2);
    ctx.stroke();
  } else if (shape.type === 'rectangle') {
    ctx.strokeRect(shape.x, shape.y, shape.w, shape.h);
  } else if (shape.type === 'line') {
    ctx.beginPath();
    ctx.moveTo(shape.start.x, shape.start.y);
    ctx.lineTo(shape.end.x, shape.end.y);
    ctx.stroke();
  }
}

// Drawing overlays and tools
function drawSkeleton(ctx, landmarks) {
  ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
  ctx.lineWidth = 2;
  HAND_CONNECTIONS.forEach(([i, j]) => {
    const p1 = landmarks[i], p2 = landmarks[j];
    ctx.beginPath();
    // Re-flip for mirrored display on UI canvas
    ctx.moveTo((1 - p1.x) * ctx.canvas.width, p1.y * ctx.canvas.height);
    ctx.lineTo((1 - p2.x) * ctx.canvas.width, p2.y * ctx.canvas.height);
    ctx.stroke();
  });
  
  ctx.fillStyle = 'red';
  landmarks.forEach(p => {
    ctx.beginPath();
    ctx.arc((1 - p.x) * ctx.canvas.width, p.y * ctx.canvas.height, 4, 0, 2 * Math.PI);
    ctx.fill();
  });
}

function drawStatsAndMode(ctx, mode) {
  const timeStr = stats.startTime ? Math.floor((Date.now() - stats.startTime) / 1000) + 's' : '0s';
  const text = `Mode: ${mode} | Strokes: ${stats.strokeCount} | Colors: ${stats.colorsUsed.size} | Shapes: ${stats.shapesRecognized} | Time: ${timeStr}`;
  
  ctx.fillStyle = 'rgba(0,0,0,0.5)';
  ctx.fillRect(10, ctx.canvas.height - 40, ctx.canvas.width - 20, 30);
  
  ctx.fillStyle = 'white';
  ctx.font = '16px Arial';
  ctx.fillText(text, 20, ctx.canvas.height - 20);
}

// Main Process Loop
let wasDrawing = false;

async function processFrame() {
  if (!isRunning || !video.videoWidth) {
    if (isRunning) requestAnimationFrame(processFrame);
    return;
  }

  // Draw Mirrored Video
  videoCtx.save();
  videoCtx.scale(-1, 1);
  videoCtx.translate(-videoCanvas.width, 0);
  videoCtx.drawImage(video, 0, 0, videoCanvas.width, videoCanvas.height);
  videoCtx.restore();

  // Run MediaPipe
  let results;
  if (handLandmarker) {
    results = handLandmarker.detectForVideo(video, performance.now());
  }
  
  uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
  
  let currentGesture = 'IDLE';

  if (results && results.landmarks && results.landmarks.length > 0) {
    // Process first hand
    const landmarks = results.landmarks[0];
    
    // Draw skeleton
    drawSkeleton(uiCtx, landmarks);
    
    // Get Index Finger tip (8)
    let idxX = landmarks[8].x * drawCanvas.width;
    let idxY = landmarks[8].y * drawCanvas.height;
    
    // Mirrored coordinates mapping
    idxX = drawCanvas.width - idxX;
    
    // Smoothing
    const smoothed = smoother.update(idxX, idxY);
    idxX = smoothed.x;
    idxY = smoothed.y;
    
    // Determine gesture
    const fingers = countFingersUp(landmarks);
    currentGesture = getGesture(fingers);
    gestureMode = currentGesture;
    
    // Draw Cursor
    uiCtx.beginPath();
    uiCtx.arc(idxX, idxY, isEraser ? 20 : 5, 0, 2 * Math.PI);
    uiCtx.fillStyle = isEraser ? 'rgba(255,255,255,0.5)' : currentColor;
    uiCtx.fill();
    uiCtx.lineWidth = 2;
    uiCtx.strokeStyle = 'white';
    uiCtx.stroke();
    
    const now = Date.now();
    
    if (currentGesture === 'DRAW') {
      if (!wasDrawing) {
        saveState(); // new stroke begins
        xp = idxX;
        yp = idxY;
        wasDrawing = true;
        currentStroke = [];
        stats.strokeCount++;
        if (!stats.startTime) stats.startTime = Date.now();
      }
      
      const dist = Math.sqrt((idxX - xp)**2 + (idxY - yp)**2);
      if (dist >= 3) {
        drawCtx.beginPath();
        drawCtx.moveTo(xp, yp);
        drawCtx.lineTo(idxX, idxY);
        drawCtx.strokeStyle = isEraser ? 'rgba(0,0,0,1)' : currentColor;
        drawCtx.lineWidth = isEraser ? 40 : currentThickness;
        drawCtx.lineCap = 'round';
        drawCtx.lineJoin = 'round';
        if (isEraser) {
          drawCtx.globalCompositeOperation = 'destination-out';
          drawCtx.stroke();
          drawCtx.globalCompositeOperation = 'source-over';
        } else {
          drawCtx.stroke();
        }
        
        currentStroke.push({x: idxX, y: idxY});
        xp = idxX;
        yp = idxY;
      }
    } else {
      // Finished Drawing a stroke
      if (wasDrawing && shapeMode && currentStroke.length > 0 && !isEraser) {
        // Clear stroke and draw perfect shape
        performUndo(); // remove rough stroke
        saveState(); // save empty state for redo if shape drawn
        const shape = recognizeShape(currentStroke);
        if (shape) {
          drawPerfectShape(drawCtx, shape, currentColor, currentThickness);
          stats.shapesRecognized++;
        }
      }
      wasDrawing = false;
    }
    
    if (currentGesture === 'UNDO' && now - lastUndoTime > COOLDOWN) {
      performUndo();
      lastUndoTime = now;
    }
    if (currentGesture === 'REDO' && now - lastRedoTime > COOLDOWN) {
      performRedo();
      lastRedoTime = now;
    }
    
  } else {
    wasDrawing = false;
  }
  
  drawStatsAndMode(uiCtx, currentGesture);
  
  animationFrameId = requestAnimationFrame(processFrame);
}

// Setup UI Handlers
function setupUI() {
  const startBtn = document.getElementById('startCameraBtn');
  const stopBtn = document.getElementById('stopCameraBtn');
  const clearBtn = document.getElementById('clearBtn');
  const undoBtn = document.getElementById('undoBtn');
  const redoBtn = document.getElementById('redoBtn');
  const shapeBtn = document.getElementById('shapeBtn');
  const saveBtn = document.getElementById('saveBtn');
  const eraserBtn = document.getElementById('eraserBtn');
  const colorButtons = document.querySelectorAll('.color-btn');
  
  if (startBtn) {
    startBtn.addEventListener('click', async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
        video.srcObject = stream;
        video.play();
        isRunning = true;
        stats.startTime = Date.now();
        if (startBtn) startBtn.style.display = 'none';
        if (stopBtn) stopBtn.style.display = 'inline-block';
        if (!handLandmarker) await initMediaPipe();
        processFrame();
      } catch (err) {
        console.error("Camera access denied or failed", err);
        alert("Could not access camera. Please allow camera permissions.");
      }
    });
  }
  
  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      isRunning = false;
      if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
      }
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
      uiCtx.clearRect(0, 0, uiCanvas.width, uiCanvas.height);
      videoCtx.clearRect(0, 0, videoCanvas.width, videoCanvas.height);
      
      if (stopBtn) stopBtn.style.display = 'none';
      if (startBtn) startBtn.style.display = 'inline-block';
    });
  }

  if (clearBtn) clearBtn.addEventListener('click', () => {
    saveState();
    drawCtx.clearRect(0, 0, drawCanvas.width, drawCanvas.height);
  });
  
  if (undoBtn) undoBtn.addEventListener('click', () => performUndo());
  if (redoBtn) redoBtn.addEventListener('click', () => performRedo());
  
  if (shapeBtn) {
    shapeBtn.addEventListener('click', () => {
      shapeMode = !shapeMode;
      shapeBtn.classList.toggle('active', shapeMode);
    });
  }
  
  if (eraserBtn) {
    eraserBtn.addEventListener('click', () => {
      isEraser = !isEraser;
      eraserBtn.classList.toggle('active', isEraser);
    });
  }
  
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      const link = document.createElement('a');
      link.download = 'my-drawing.png';
      link.href = drawCanvas.toDataURL('image/png');
      link.click();
    });
  }
  
  if (colorButtons) {
    colorButtons.forEach(btn => {
      btn.addEventListener('click', (e) => {
        isEraser = false;
        if (eraserBtn) eraserBtn.classList.remove('active');
        
        colorButtons.forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        
        currentColor = e.target.dataset.color || e.target.style.backgroundColor;
        stats.colorsUsed.add(currentColor);
      });
    });
  }
}

// Initialization
document.addEventListener('DOMContentLoaded', () => {
  initParticles();
  setupCanvases();
  setupUI();
  initMediaPipe(); // pre-load
});
