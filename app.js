// app.js — face detection + recognition UI logic
// Requires face-api.js (included in index.html via CDN)
// Make sure models are placed in /models and server is running via http(s)

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const overlayCtx = overlay.getContext('2d');

const btnLoadModels = document.getElementById('btnLoadModels');
const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');

const modelsStatus = document.getElementById('modelsStatus');
const faceCountEl = document.getElementById('faceCount');
const expressionsEl = document.getElementById('expressions');
const ageEl = document.getElementById('age');
const genderEl = document.getElementById('gender');
const identifiedList = document.getElementById('identifiedList');
const dbCount = document.getElementById('dbCount');

const detectionModelSelect = document.getElementById('detectionModel');
const inputSizeEl = document.getElementById('inputSize');
const scoreThresholdEl = document.getElementById('scoreThreshold');

const labelNameInput = document.getElementById('labelName');
const labeledImagesInput = document.getElementById('labeledImages');
const btnRegister = document.getElementById('btnRegister');
const btnClearDB = document.getElementById('btnClearDB');
const btnExport = document.getElementById('btnExport');
const btnImport = document.getElementById('btnImport');
const importFileInput = document.getElementById('importFile');

let stream = null;
let detectionOptions;
let isModelsLoaded = false;
let running = false;
let faceMatcher = null; // will be created after loading labeled descriptors

// Key for localStorage
const LS_DB_KEY = 'facelab_db_v1';

// utility: load models from /models
async function loadModels() {
  modelsStatus.innerText = 'loading...';
  try {
    // load lightweight/tiny + landmark + recognition + age/gender + expressions
    await faceapi.nets.tinyFaceDetector.loadFromUri('/models');
    await faceapi.nets.ssdMobilenetv1.loadFromUri('/models'); // optional for higher accuracy
    await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
    await faceapi.nets.faceRecognitionNet.loadFromUri('/models');
    await faceapi.nets.ageGenderNet.loadFromUri('/models');
    await faceapi.nets.faceExpressionNet.loadFromUri('/models');

    isModelsLoaded = true;
    modelsStatus.innerText = 'loaded ✓';
    makeFaceMatcherFromStorage();
  } catch (err) {
    console.error(err);
    modelsStatus.innerText = 'error loading models — check /models';
  }
}

btnLoadModels.addEventListener('click', loadModels);

// Start camera
btnStart.addEventListener('click', async () => {
  if (!isModelsLoaded) {
    await loadModels();
  }
  startVideo();
});

btnStop.addEventListener('click', stopVideo);

// setup detection options
function updateDetectionOptions() {
  const model = detectionModelSelect.value;
  const inputSize = parseInt(inputSizeEl.value) || 320;
  const scoreThreshold = parseFloat(scoreThresholdEl.value) || 0.5;

  if (model === 'tiny') {
    detectionOptions = new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold });
  } else {
    detectionOptions = new faceapi.SsdMobilenetv1Options({ minConfidence: scoreThreshold });
  }
}

detectionModelSelect.addEventListener('change', updateDetectionOptions);
inputSizeEl.addEventListener('change', updateDetectionOptions);
scoreThresholdEl.addEventListener('change', updateDetectionOptions);

updateDetectionOptions();

async function startVideo() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    running = true;
    detectionLoop();
  } catch (err) {
    alert('Camera access denied or not available. Check permissions and that you are using https/http://localhost.');
    console.error(err);
  }
}

function stopVideo() {
  running = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
}

// continuous detection loop
async function detectionLoop() {
  if (!running) return;
  if (!isModelsLoaded) {
    requestAnimationFrame(detectionLoop);
    return;
  }

  updateDetectionOptions();

  // run detection with multiple outputs
  const detections = await faceapi.detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
    .withFaceLandmarks()
    .withFaceExpressions()
    .withAgeAndGender()
    .withFaceDescriptors();

  // clear & draw
  const displaySize = { width: video.videoWidth, height: video.videoHeight };
  faceapi.matchDimensions(overlay, displaySize);

  const resized = faceapi.resizeResults(detections, displaySize);

  // clear overlay and prepare identifications for this frame
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  const identifications = [];

  for (const det of resized) {
    const box = det.detection.box;

    // draw box
    overlayCtx.strokeStyle = 'rgba(42,157,143,0.9)';
    overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(box.x, box.y, box.width, box.height);

    // draw landmarks
    const lm = det.landmarks.positions;
    overlayCtx.fillStyle = 'rgba(255,255,255,0.9)';
    for (const pt of lm) {
      overlayCtx.beginPath();
      overlayCtx.arc(pt.x, pt.y, 2, 0, Math.PI * 2);
      overlayCtx.fill();
    }

    // expressions (pick max)
    const expr = det.expressions || {};
    const maxExp = Object.entries(expr).sort((a, b) => b[1] - a[1])[0];
    const expText = maxExp ? `${maxExp[0]} (${Math.round(maxExp[1] * 100)}%)` : '—';
    expressionsEl.innerText = expText;

    // age & gender
    const ageVal = det.age ? det.age.toFixed(0) : '—';
    const genderVal = det.gender ? `${det.gender} (${Math.round(det.genderProbability * 100)}%)` : '—';
    ageEl.innerText = ageVal;
    genderEl.innerText = genderVal;

    // identification
    if (faceMatcher && det.descriptor) {
      const best = faceMatcher.findBestMatch(det.descriptor);
      if (best && best.label !== 'unknown') {
        identifications.push({ name: best.label, distance: best.distance, box });
        // draw label
        overlayCtx.fillStyle = 'rgba(42,157,143,0.95)';
        overlayCtx.font = '16px Inter, sans-serif';
        overlayCtx.fillText(`${best.label} (${best.distance.toFixed(2)})`, box.x + 6, box.y - 8);
      } else {
        overlayCtx.fillStyle = 'rgba(255,140,90,0.9)';
        overlayCtx.font = '14px Inter, sans-serif';
        overlayCtx.fillText('unknown', box.x + 6, box.y - 8);
      }
    }
  }

  // update identified list
  identifiedList.innerHTML = '';
  identifications.forEach(id => {
    const li = document.createElement('li');
    li.textContent = `${id.name} — dist ${id.distance.toFixed(2)}`;
    identifiedList.appendChild(li);
  });

  requestAnimationFrame(detectionLoop);
}

// -- Registration / labeled face DB --

async function registerLabeledFaces() {
  const files = Array.from(labeledImagesInput.files || []);
  const label = (labelNameInput.value || '').trim();
  if (!label) { alert('Enter a name to register'); return; }
  if (files.length === 0) { alert('Choose one or more images for this person'); return; }
  if (!isModelsLoaded) { alert('Load models first'); return; }

  const descriptors = [];
  for (const f of files) {
    const img = await faceapi.bufferToImage(f);
    // detect single face in provided image
    const detection = await faceapi.detectSingleFace(img, new faceapi.TinyFaceDetectorOptions()).withFaceLandmarks().withFaceDescriptor();
    if (!detection || !detection.descriptor) {
      console.warn('No face found in', f.name);
      continue;
    }
    descriptors.push(Array.from(detection.descriptor));
  }

  if (descriptors.length === 0) {
    alert('No faces detected in uploaded images. Try clearer photos.');
    return;
  }

  // load existing DB
  const db = loadDB();
  if (!db[label]) db[label] = [];
  db[label] = db[label].concat(descriptors);
  saveDB(db);
  makeFaceMatcherFromStorage();
  alert('Registered ${descriptors.length} face(s) for ${label}');
  labeledImagesInput.value = '';
  labelNameInput.value = '';
}

btnRegister.addEventListener('click', registerLabeledFaces);
btnClearDB.addEventListener('click', () => {
  if (!confirm('Clear all registered faces?')) return;
  localStorage.removeItem(LS_DB_KEY);
  makeFaceMatcherFromStorage();
});

// persistence utilities
function saveDB(db) {
  localStorage.setItem(LS_DB_KEY, JSON.stringify(db));
}

function loadDB() {
  try {
    return JSON.parse(localStorage.getItem(LS_DB_KEY) || '{}');
  } catch (e) {
    return {};
  }
}

// create faceMatcher from stored descriptors
function makeFaceMatcherFromStorage() {
  const db = loadDB();
  const labeledDescriptors = [];
  let count = 0;
  for (const label of Object.keys(db)) {
    const descriptorArrays = db[label];
    const floatDescs = descriptorArrays.map(d => new Float32Array(d));
    labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, floatDescs));
    count++;
  }
  dbCount.innerText = count;
  faceMatcher = labeledDescriptors.length ? new faceapi.FaceMatcher(labeledDescriptors, 0.6) : null;
}

// export/import
btnExport.addEventListener('click', () => {
  const db = loadDB();
  const blob = new Blob([JSON.stringify(db)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'facelab_db.json'; a.click();
  URL.revokeObjectURL(url);
});

btnImport.addEventListener('click', () => importFileInput.click());
importFileInput.addEventListener('change', async (ev) => {
  const f = ev.target.files[0];
  if (!f) return;
  const text = await f.text();
  try {
    const parsed = JSON.parse(text);
    localStorage.setItem(LS_DB_KEY, JSON.stringify(parsed));
    makeFaceMatcherFromStorage();
    alert('Imported DB successfully');
  } catch (e) {
    alert('Invalid JSON file');
  }
});

// initialize on load
window.addEventListener('DOMContentLoaded', () => {
  if (navigator.mediaDevices === undefined) {
    alert('Your browser does not support camera access. Use latest Chrome/Edge/Firefox/Safari.');
  }
  makeFaceMatcherFromStorage();
  // optionally auto-load models for convenience (comment if you want manual)
  // loadModels();
});