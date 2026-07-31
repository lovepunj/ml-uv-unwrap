import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';

// Backend base URL. Override via config.js: window.UVUNWRAP_API = 'https://...';
const API = (typeof window !== 'undefined' && window.UVUNWRAP_API) || '';
window.__mluvLoaded = true;
console.log('[ML-UV] app.js module loaded successfully');

// ─── State ───
let currentJobId = null;
let currentView = '3d'; // '3d' | 'uv'
let pollTimer = null;
let seamLines = null;
let measurePoints = [];
let measureLine = null;
let measureMode = false;
let normalsHelper = null;
let normalsVisible = false;
let seamGraphLines = null;
const originalPositions = new Map();
const originalMeshMaterials = new Map();
let pointCloudMode = false;
let pointCloudObjects = [];
let dualPaneEnabled = false;
let uvRenderer = null;
let clippingEnabled = false;
let explodeEnabled = false;
let meshAnalysisCache = null;

// ─── Cut/Join State ───
let cutMode = false;
let joinMode = false;
let cutEdgeData = null;
let selectedCutEdges = new Set();
let cutEdgeLines = null;
let uvIslandData = null;
let selectedIslands = new Set();
let islandFaceMeshes = [];
const cutRaycaster = new THREE.Raycaster();
const cutMouse = new THREE.Vector2();

// ─── Three.js Setup ───
const canvas = document.getElementById('three-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x0f1117);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, 1, 0.01, 100);
camera.position.set(2, 1.5, 2);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = false;
controls.autoRotateSpeed = 1.5;
controls.enablePan = true;
controls.panSpeed = 0.8;
controls.zoomSpeed = 1.2;
controls.rotateSpeed = 0.8;

// Lighting
const ambientLight = new THREE.AmbientLight(0x404060, 1.5);
scene.add(ambientLight);

const dirLight = new THREE.DirectionalLight(0xffffff, 2);
dirLight.position.set(3, 5, 3);
dirLight.castShadow = true;
dirLight.shadow.mapSize.width = 1024;
dirLight.shadow.mapSize.height = 1024;
scene.add(dirLight);

const rimLight = new THREE.DirectionalLight(0x6366f1, 0.8);
rimLight.position.set(-3, 2, -3);
scene.add(rimLight);

const fillLight = new THREE.DirectionalLight(0x22c55e, 0.3);
fillLight.position.set(0, -2, 0);
scene.add(fillLight);

// Grid
const gridHelper = new THREE.GridHelper(4, 20, 0x2a2e3e, 0x1c1f2e);
scene.add(gridHelper);

// Checker texture for UV visualization
const checkerTexture = createCheckerTexture();
checkerTexture.wrapS = THREE.RepeatWrapping;
checkerTexture.wrapT = THREE.RepeatWrapping;

let currentMesh = null;
let frameCount = 0;
let lastFpsTime = performance.now();

function createCheckerTexture(size = 512, checks = 16) {
  const c = document.createElement('canvas');
  c.width = c.height = size;
  const ctx = c.getContext('2d');
  const step = size / checks;
  for (let y = 0; y < checks; y++) {
    for (let x = 0; x < checks; x++) {
      ctx.fillStyle = (x + y) % 2 === 0 ? '#ddd' : '#999';
      ctx.fillRect(x * step, y * step, step, step);
    }
  }
  return new THREE.CanvasTexture(c);
}

// ─── Resize ───
function onResize() {
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth;
  const h = vp.clientHeight;
  if (w > 0 && h > 0) {
    if (dualPaneEnabled) {
      renderer.setSize(Math.floor(w / 2), h);
      camera.aspect = (w / 2) / h;
      if (uvRenderer) {
        uvRenderer.setSize(Math.floor(w / 2), h);
      }
    } else {
      renderer.setSize(w, h);
      camera.aspect = w / h;
    }
    camera.updateProjectionMatrix();
  }
}
window.addEventListener('resize', onResize);

const vp = document.getElementById('viewport');
if (vp) {
  const ro = new ResizeObserver(() => onResize());
  ro.observe(vp);
}
onResize();
requestAnimationFrame(() => onResize());

// ─── Render Loop with FPS Counter ───
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);

  // FPS counter
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTime >= 1000) {
    const fps = Math.round(frameCount * 1000 / (now - lastFpsTime));
    document.getElementById('hud-fps').textContent = fps;
    frameCount = 0;
    lastFpsTime = now;
  }
}
animate();

// ─── DOM Elements ───
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const meshInfo = document.getElementById('mesh-info');
const previewPanel = document.getElementById('preview-panel');
const unwrapBtn = document.getElementById('unwrap-btn');
const downloadBtn = document.getElementById('download-btn');
const downloadUvBtn = document.getElementById('download-uv-btn');
const progressPanel = document.getElementById('progress-panel');
const resultsPanel = document.getElementById('results-panel');
const viewportEmpty = document.getElementById('viewport-empty');
const uvOverlay = document.getElementById('uv-overlay');
const uvImage = document.getElementById('uv-image');
const progressFill = document.getElementById('progress-fill');
const progressText = document.getElementById('progress-text');
const screenshotBtn = document.getElementById('screenshot-btn');
const resetCameraBtn = document.getElementById('reset-camera-btn');
const measureOverlay = document.getElementById('measure-overlay');
const measureDistance = document.getElementById('measure-distance');
const methodSelect = document.getElementById('method-select');

// ─── Slider Bindings ───
const sliders = [
  { id: 'iterations', display: 'iter-value', format: v => v },
  { id: 'num-points', display: 'points-value', format: v => v },
  { id: 'num-charts', display: 'charts-value', format: v => v },
  { id: 'lr', display: 'lr-value', format: v => Math.pow(10, parseFloat(v)).toExponential(0) },
];

sliders.forEach(({ id, display, format }) => {
  const el = document.getElementById(id);
  const out = document.getElementById(display);
  el.addEventListener('input', () => {
    out.textContent = format(el.value);
  });
});

// ─── File Upload ───
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) uploadFile(fileInput.files[0]);
});

async function uploadFile(file) {
  // Reset state
  resetState();

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const err = await res.json();
      alert(err.detail || 'Upload failed');
      return;
    }
    const data = await res.json();
    currentJobId = data.job_id;

    // Update UI
    document.getElementById('info-filename').textContent = data.mesh_info.filename;
    document.getElementById('info-verts').textContent = data.mesh_info.vertices.toLocaleString();
    document.getElementById('info-faces').textContent = data.mesh_info.faces.toLocaleString();
    document.getElementById('info-bounds').textContent = data.mesh_info.bounds || '-';
    meshInfo.hidden = false;
    previewPanel.hidden = false;
    unwrapBtn.disabled = false;

    // Update HUD
    document.getElementById('hud-verts').textContent = data.mesh_info.vertices.toLocaleString();
    document.getElementById('hud-faces').textContent = data.mesh_info.faces.toLocaleString();

    // Load 3D preview
    loadMeshPreview(file);

  } catch (err) {
    console.error('Upload error:', err);
    alert('Upload failed: ' + err.message);
  }
}

const originalMaterials = new Map();

function loadMeshPreview(file) {
  viewportEmpty.hidden = true;
  const name = file.name.toLowerCase();

  // Remove old mesh
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh = null;
  }
  originalMaterials.clear();

  const url = URL.createObjectURL(file);

  if (name.endsWith('.glb') || name.endsWith('.gltf')) {
    new GLTFLoader().load(url, gltf => {
      currentMesh = gltf.scene;
      currentMesh.traverse(child => {
        if (child.isMesh) originalMaterials.set(child.uuid, child.material.clone());
      });
      fitCamera(currentMesh);
      scene.add(currentMesh);
      updatePreviewInfo(currentMesh);
    }, undefined, err => console.error('GLB load error:', err));
  } else {
    new OBJLoader().load(url, obj => {
      obj.traverse(child => {
        if (child.isMesh) {
          child.material = new THREE.MeshStandardMaterial({
            color: 0x8888aa,
            roughness: 0.6,
            metalness: 0.1,
          });
          originalMaterials.set(child.uuid, child.material.clone());
        }
      });
      currentMesh = obj;
      fitCamera(currentMesh);
      scene.add(currentMesh);
      updatePreviewInfo(currentMesh);
    }, undefined, err => console.error('OBJ load error:', err));
  }
}

function updatePreviewInfo(obj) {
  let totalVerts = 0, totalFaces = 0, totalEdges = 0;
  obj.traverse(child => {
    if (child.isMesh && child.geometry) {
      const geo = child.geometry;
      totalVerts += geo.attributes.position ? geo.attributes.position.count : 0;
      totalFaces += geo.index ? geo.index.count / 3 : (geo.attributes.position ? geo.attributes.position.count / 3 : 0);
      if (geo.index) {
        totalEdges += geo.index.count / 2;
      }
    }
  });
  document.getElementById('prev-faces').textContent = Math.round(totalFaces).toLocaleString();
  document.getElementById('prev-edges').textContent = Math.round(totalEdges).toLocaleString();
}

function fitCamera(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const dist = maxDim * 2;

  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(dist * 0.6, dist * 0.4, dist * 0.6));
  camera.lookAt(center);
  controls.update();
}

// ─── Unwrap ───
unwrapBtn.addEventListener('click', startUnwrap);

async function startUnwrap() {
  if (!currentJobId) return;

  // Clear previous result but keep uploaded mesh
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh = null;
  }
  clearSeamLines();
  clearMeasurements();
  uvOverlay.hidden = true;
  viewportEmpty.hidden = true;
  setView('3d');

  unwrapBtn.disabled = true;
  downloadBtn.hidden = true;
  downloadUvBtn.hidden = true;
  resultsPanel.hidden = true;
  progressPanel.hidden = false;
  progressFill.style.width = '0%';

  const classicalMethods = ['xatlas', 'lscm', 'abf', 'arap', 'harmonic', 'conformal', 'graph_cuts', 'hilbert'];
  const selectedMethod = methodSelect.value;

  progressText.textContent = `Starting ${selectedMethod}...`;

  const lr = Math.pow(10, parseFloat(document.getElementById('lr').value));
  const form = new FormData();

  let endpoint;

  if (selectedMethod.startsWith('classical-')) {
    const method = selectedMethod.replace('classical-', '');
    form.append('method', method);
    form.append('max_charts', '0');
    endpoint = 'unwrap-classical';
  } else if (selectedMethod === 'partuv') {
    form.append('num_iterations', document.getElementById('iterations').value);
    form.append('num_points', document.getElementById('num-points').value);
    form.append('num_charts', document.getElementById('num-charts').value);
    form.append('lr', lr.toString());
    form.append('mode', 'partuv');
    endpoint = 'unwrap';
  } else if (selectedMethod === 'multi_chart') {
    form.append('num_iterations', document.getElementById('iterations').value);
    form.append('num_points', document.getElementById('num-points').value);
    form.append('num_charts', document.getElementById('num-charts').value);
    form.append('lr', lr.toString());
    form.append('mode', 'multi_chart');
    endpoint = 'unwrap';
  } else if (selectedMethod === 'detect') {
    form.append('num_iterations', document.getElementById('iterations').value);
    form.append('num_points', document.getElementById('num-points').value);
    form.append('num_charts', document.getElementById('num-charts').value);
    form.append('lr', lr.toString());
    endpoint = 'detect';
  } else if (selectedMethod === 'hybrid') {
    form.append('num_iterations', document.getElementById('iterations').value);
    form.append('num_points', document.getElementById('num-points').value);
    form.append('num_charts', document.getElementById('num-charts').value);
    form.append('lr', lr.toString());
    form.append('mode', 'hybrid');
    form.append('classical_method', 'xatlas');
    endpoint = 'unwrap';
  } else {
    form.append('num_iterations', document.getElementById('iterations').value);
    form.append('num_points', document.getElementById('num-points').value);
    form.append('num_charts', document.getElementById('num-charts').value);
    form.append('lr', lr.toString());
    form.append('mode', selectedMethod);
    endpoint = 'unwrap';
  }

  try {
    const url = `${API}/api/${endpoint}/${currentJobId}`;
    console.log('Unwrap request:', url, Object.fromEntries(form.entries()));
    const res = await fetch(url, {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      const err = await res.json();
      progressText.textContent = `Error: ${err.detail || 'Failed to start'}`;
      unwrapBtn.disabled = false;
      return;
    }

    pollStatus();

  } catch (err) {
    console.error('Unwrap error:', err);
    progressText.textContent = `Error: ${err.message}`;
    unwrapBtn.disabled = false;
  }
}

function pollStatus() {
  if (pollTimer) clearInterval(pollTimer);

  pollTimer = setInterval(async () => {
    try {
      const res = await fetch(`${API}/api/status/${currentJobId}`);
      const data = await res.json();

      progressFill.style.width = `${data.progress}%`;

      if (data.status === 'processing') {
        const mode = data.use_partuv ? 'PartUV' : 'Standard';
        progressText.textContent = `${mode} processing... ${data.progress}%`;
      } else if (data.status === 'completed') {
        clearInterval(pollTimer);
        pollTimer = null;
        onUnwrapComplete(data);
      } else if (data.status === 'failed') {
        clearInterval(pollTimer);
        pollTimer = null;
        progressText.textContent = `Failed: ${data.error}`;
        unwrapBtn.disabled = false;
      }
    } catch (err) {
      console.error('Poll error:', err);
    }
  }, 500);
}

function onUnwrapComplete(statusData) {
  progressPanel.hidden = true;
  downloadBtn.hidden = false;
  resultsPanel.hidden = false;
  unwrapBtn.disabled = false;

  // Clear analysis cache for new result
  meshAnalysisCache = null;

  if (statusData.uv_stats) {
    document.getElementById('stat-uv-verts').textContent =
      statusData.uv_stats.num_verts.toLocaleString();
  }

  const partStats = document.getElementById('partuv-stats');
  if (partStats && statusData.use_partuv) {
    partStats.hidden = false;
    document.getElementById('stat-parts').textContent =
      statusData.num_parts?.toString() || 'N/A';
  } else if (partStats) {
    partStats.hidden = true;
  }

  loadResultMesh();

  const uvSize = document.getElementById('uv-size').value;
  uvImage.src = `${API}/api/uv-image/${currentJobId}?size=${uvSize}&t=${Date.now()}`;
  downloadUvBtn.hidden = false;

  showUVPreview();
}

function loadResultMesh() {
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh = null;
  }
  clearSeamLines();
  originalMaterials.clear();

  const ts = Date.now();
  const url = `${API}/api/preview/${currentJobId}?t=${ts}`;

  new GLTFLoader().load(url, gltf => {
    currentMesh = gltf.scene;
    currentMesh.traverse(child => {
      if (child.isMesh) originalMaterials.set(child.uuid, child.material.clone());
    });
    applyMaterial(currentMaterial);
    fitCamera(currentMesh);
    scene.add(currentMesh);
    updatePreviewInfo(currentMesh);
    showSeamLines();
  }, undefined, () => {
    new OBJLoader().load(url, obj => {
      currentMesh = obj;
      currentMesh.traverse(child => {
        if (child.isMesh) {
          originalMaterials.set(child.uuid, child.material.clone());
        }
      });
      applyMaterial(currentMaterial);
      fitCamera(currentMesh);
      scene.add(currentMesh);
      updatePreviewInfo(currentMesh);
      showSeamLines();
    });
  });
}

// ─── Seam Visualization ───
function showSeamLines() {
  if (!currentMesh || !currentJobId) return;

  clearSeamLines();

  const lineMaterial = new THREE.LineBasicMaterial({
    color: 0xef4444,
    linewidth: 2,
    transparent: true,
    opacity: 0.8,
  });

  currentMesh.traverse(child => {
    if (!child.isMesh || !child.geometry) return;
    const geo = child.geometry;
    if (!geo.index) return;

    const positions = geo.attributes.position;
    const index = geo.index;
    const edges = new Map();

    for (let i = 0; i < index.count; i += 3) {
      const a = index.getX(i);
      const b = index.getX(i + 1);
      const c = index.getX(i + 2);
      addEdge(edges, a, b);
      addEdge(edges, b, c);
      addEdge(edges, a, c);
    }

    const seamVerts = [];
    for (const [key, count] of edges) {
      if (count === 1) {
        const [v0, v1] = key.split('-').map(Number);
        seamVerts.push(
          positions.getX(v0), positions.getY(v0), positions.getZ(v0),
          positions.getX(v1), positions.getY(v1), positions.getZ(v1),
        );
      }
    }

    if (seamVerts.length > 0) {
      const seamGeo = new THREE.BufferGeometry();
      seamGeo.setAttribute('position', new THREE.Float32BufferAttribute(seamVerts, 3));
      const lines = new THREE.LineSegments(seamGeo, lineMaterial);
      if (!seamLines) seamLines = new THREE.Group();
      seamLines.add(lines);
    }
  });

  if (seamLines) scene.add(seamLines);
}

function addEdge(map, a, b) {
  const key = a < b ? `${a}-${b}` : `${b}-${a}`;
  map.set(key, (map.get(key) || 0) + 1);
}

function clearSeamLines() {
  if (seamLines) {
    scene.remove(seamLines);
    seamLines.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    seamLines = null;
  }
}

document.getElementById('seam-toggle').addEventListener('click', function() {
  this.classList.toggle('active');
  if (seamLines) {
    seamLines.visible = this.classList.contains('active');
  } else if (this.classList.contains('active')) {
    showSeamLines();
  }
});

// ─── Measurement Tool ───
const measureMaterial = new THREE.LineBasicMaterial({ color: 0xeab308, linewidth: 2 });
const measurePointMaterial = new THREE.MeshBasicMaterial({ color: 0xeab308 });
const measurePointGeo = new THREE.SphereGeometry(0.02, 8, 8);

document.getElementById('measure-toggle').addEventListener('click', function() {
  measureMode = !measureMode;
  this.classList.toggle('active', measureMode);
  measureOverlay.hidden = !measureMode;
  if (!measureMode) clearMeasurements();
  controls.enableRotate = !measureMode;
});

canvas.addEventListener('click', e => {
  if (!measureMode || !currentMesh) return;

  const rect = canvas.getBoundingClientRect();
  const mouse = new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  );

  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(mouse, camera);
  const intersects = raycaster.intersectObject(currentMesh, true);

  if (intersects.length > 0) {
    const pt = intersects[0].point;
    const marker = new THREE.Mesh(measurePointGeo, measurePointMaterial);
    marker.position.copy(pt);
    scene.add(marker);
    measurePoints.push({ point: pt.clone(), marker });

    if (measurePoints.length === 2) {
      const p0 = measurePoints[0].point;
      const p1 = measurePoints[1].point;
      const dist = p0.distanceTo(p1);

      const lineGeo = new THREE.BufferGeometry().setFromPoints([p0, p1]);
      measureLine = new THREE.Line(lineGeo, measureMaterial);
      scene.add(measureLine);

      measureDistance.textContent = `Distance: ${dist.toFixed(4)} units`;
    } else if (measurePoints.length > 2) {
      clearMeasurements();
      measurePoints.push({ point: pt.clone(), marker });
    }
  }
});

function clearMeasurements() {
  measurePoints.forEach(p => scene.remove(p.marker));
  measurePoints = [];
  if (measureLine) {
    scene.remove(measureLine);
    measureLine = null;
  }
  measureDistance.textContent = 'Click two points to measure';
}

// ─── Vertex Normals Toggle ───
document.getElementById('normals-toggle').addEventListener('click', function() {
  normalsVisible = !normalsVisible;
  this.classList.toggle('active', normalsVisible);
  if (normalsVisible) {
    showNormals();
  } else {
    hideNormals();
  }
});

function showNormals() {
  if (!currentMesh) return;
  hideNormals();
  normalsHelper = new THREE.Group();

  const normalMaterial = new THREE.LineBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.5 });

  currentMesh.traverse(child => {
    if (!child.isMesh || !child.geometry) return;
    const geo = child.geometry;
    if (!geo.attributes.position) return;

    const positions = geo.attributes.position;
    const normalAttr = geo.attributes.normal;
    if (!normalAttr) return;

    const lineVerts = [];
    const normalLen = 0.05;

    for (let i = 0; i < Math.min(positions.count, 2000); i++) {
      const x = positions.getX(i);
      const y = positions.getY(i);
      const z = positions.getZ(i);
      const nx = normalAttr.getX(i);
      const ny = normalAttr.getY(i);
      const nz = normalAttr.getZ(i);
      lineVerts.push(x, y, z, x + nx * normalLen, y + ny * normalLen, z + nz * normalLen);
    }

    if (lineVerts.length > 0) {
      const lineGeo = new THREE.BufferGeometry();
      lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(lineVerts, 3));
      normalsHelper.add(new THREE.LineSegments(lineGeo, normalMaterial));
    }
  });

  scene.add(normalsHelper);
}

function hideNormals() {
  if (normalsHelper) {
    scene.remove(normalsHelper);
    normalsHelper.traverse(child => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
    });
    normalsHelper = null;
  }
}

// ─── Download ───
downloadBtn.addEventListener('click', () => {
  if (!currentJobId) return;
  window.open(`${API}/api/download/${currentJobId}`, '_blank');
});

downloadUvBtn.addEventListener('click', () => {
  if (!currentJobId) return;
  const uvSize = document.getElementById('uv-size').value;
  window.open(`${API}/api/uv-image/${currentJobId}?size=${uvSize}&download=1`, '_blank');
});

document.getElementById('uv-size').addEventListener('change', () => {
  if (!currentJobId) return;
  const uvSize = document.getElementById('uv-size').value;
  uvImage.src = `${API}/api/uv-image/${currentJobId}?size=${uvSize}&t=${Date.now()}`;
});

// ─── View Toggle ───
document.getElementById('toggle-3d').addEventListener('click', () => setView('3d'));
document.getElementById('toggle-uv').addEventListener('click', () => setView('uv'));

function setView(view) {
  currentView = view;
  document.getElementById('toggle-3d').classList.toggle('active', view === '3d');
  document.getElementById('toggle-uv').classList.toggle('active', view === 'uv');
  if (dualPaneEnabled) {
    uvOverlay.hidden = true;
  } else {
    uvOverlay.hidden = view !== 'uv';
  }
  if (view === 'uv' && currentJobId) {
    renderUVPane();
  }
}

// ─── Material Toggles ───
let currentMaterial = 'solid';
const matBtns = {
  default: document.getElementById('mat-default'),
  solid: document.getElementById('mat-solid'),
  wireframe: document.getElementById('mat-wireframe'),
  checker: document.getElementById('mat-checker'),
  normal: document.getElementById('mat-normal'),
  ao: document.getElementById('mat-ao'),
  distortion: document.getElementById('mat-distortion'),
  ao_heat: document.getElementById('mat-ao-heat'),
  curvature: document.getElementById('mat-curvature'),
  chart: document.getElementById('mat-chart'),
};

Object.entries(matBtns).forEach(([name, btn]) => {
  btn.addEventListener('click', () => {
    currentMaterial = name;
    Object.values(matBtns).forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    applyMaterial(name);
  });
});

document.getElementById('mat-texture').addEventListener('click', () => {
  document.getElementById('texture-input').click();
});

document.getElementById('mat-texture-toggle').addEventListener('click', () => {
  if (!uploadedTexture) return;
  currentMaterial = 'texture';
  Object.values(matBtns).forEach(b => b.classList.remove('active'));
  document.getElementById('mat-texture').classList.add('active');
  applyMaterial('texture');
});

const textureLoader = new THREE.TextureLoader();
let uploadedTexture = null;

document.getElementById('texture-input').addEventListener('change', function(e) {
  const file = e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  textureLoader.load(url, tex => {
    uploadedTexture = tex;
    tex.wrapS = THREE.RepeatWrapping;
    tex.wrapT = THREE.RepeatWrapping;
    document.getElementById('mat-texture-toggle').hidden = false;
    currentMaterial = 'texture';
    Object.values(matBtns).forEach(b => b.classList.remove('active'));
    matBtns.texture.classList.add('active');
    applyMaterial('texture');
  });
  this.value = '';
});

// ─── Mesh Analysis Cache ───
async function fetchMeshAnalysis() {
  if (meshAnalysisCache) return meshAnalysisCache;
  if (!currentJobId) return null;
  try {
    const res = await fetch(`${API}/api/mesh-analysis/${currentJobId}`);
    if (!res.ok) return null;
    meshAnalysisCache = await res.json();
    return meshAnalysisCache;
  } catch (e) {
    console.error('Analysis fetch failed:', e);
    return null;
  }
}

// ─── Color Utilities ───
function jetColormap(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.min(255, Math.max(0, Math.round(t < 0.5 ? 0 : t < 0.75 ? (t - 0.5) * 4 * 255 : 255)));
  const g = Math.min(255, Math.max(0, Math.round(t < 0.25 ? t * 4 * 255 : t < 0.75 ? 255 : (1 - t) * 4 * 255)));
  const b = Math.min(255, Math.max(0, Math.round(t < 0.25 ? 255 : t < 0.5 ? (1 - (t - 0.25) * 4) * 255 : 0)));
  return new THREE.Color(r / 255, g / 255, b / 255);
}

function viridisColormap(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.round(68 + t * (253 - 68));
  const g = Math.round(1 + t * (231 - 1));
  const b = Math.round(84 + t * (37 - 84));
  return new THREE.Color(r / 255, g / 255, b / 255);
}

function showLegend(title, min, max) {
  const legend = document.getElementById('color-legend');
  legend.hidden = false;
  document.getElementById('legend-title').textContent = title;
  document.getElementById('legend-min').textContent = min;
  document.getElementById('legend-max').textContent = max;
}

function hideLegend() {
  document.getElementById('color-legend').hidden = true;
}

// ─── Per-Face Coloring ───
async function applyPerFaceColoring(mode) {
  if (!currentMesh) return;
  const data = await fetchMeshAnalysis();
  if (!data) return;

  const values = mode === 'distortion' ? data.distortion :
                 mode === 'ao_heat' ? data.ao :
                 mode === 'curvature' ? data.curvature : null;
  if (!values) return;

  const minVal = Math.min(...values);
  const maxVal = Math.max(...values) || 1;
  const range = maxVal - minVal || 1;

  const titles = { distortion: 'UV Distortion', ao_heat: 'Ambient Occlusion', curvature: 'Curvature' };
  const labels = { distortion: [minVal.toFixed(3), maxVal.toFixed(3)], ao_heat: [minVal.toFixed(3), maxVal.toFixed(3)], curvature: [minVal.toFixed(3), maxVal.toFixed(3)] };
  showLegend(titles[mode], labels[mode][0], labels[mode][1]);

  currentMesh.traverse(child => {
    if (!child.isMesh || !child.geometry) return;
    const geo = child.geometry;
    const count = geo.attributes.position ? geo.attributes.position.count : 0;
    const colors = new Float32Array(count * 3);

    if (geo.index) {
      const index = geo.index;
      for (let f = 0; f < index.count / 3; f++) {
        const fi = Math.min(f, values.length - 1);
        const t = (values[fi] - minVal) / range;
        const c = viridisColormap(t);
        for (let j = 0; j < 3; j++) {
          const vi = index.getX(f * 3 + j);
          colors[vi * 3] = c.r;
          colors[vi * 3 + 1] = c.g;
          colors[vi * 3 + 2] = c.b;
        }
      }
    }

    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    child.material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.7,
      metalness: 0.0,
    });
  });
}

// ─── Chart Coloring ───
async function applyChartColoring() {
  if (!currentMesh) return;
  const data = await fetchMeshAnalysis();
  if (!data || !data.chart_labels) return;

  const chartColors = [
    new THREE.Color(0x6366f1), new THREE.Color(0x22c55e), new THREE.Color(0xef4444),
    new THREE.Color(0xeab308), new THREE.Color(0x06b6d4), new THREE.Color(0xf97316),
    new THREE.Color(0xa855f7), new THREE.Color(0xec4899), new THREE.Color(0x14b8a6),
    new THREE.Color(0x84cc16), new THREE.Color(0xf43f5e), new THREE.Color(0x8b5cf6),
  ];

  const labels = data.chart_labels;
  const numCharts = Math.max(...labels) + 1;
  showLegend('UV Charts', '0', `${numCharts - 1}`);

  currentMesh.traverse(child => {
    if (!child.isMesh || !child.geometry) return;
    const geo = child.geometry;
    const count = geo.attributes.position ? geo.attributes.position.count : 0;
    const colors = new Float32Array(count * 3);

    for (let vi = 0; vi < count && vi < labels.length; vi++) {
      const c = chartColors[labels[vi] % chartColors.length];
      colors[vi * 3] = c.r;
      colors[vi * 3 + 1] = c.g;
      colors[vi * 3 + 2] = c.b;
    }

    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    child.material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.6,
      metalness: 0.1,
    });
  });
}

function applyMaterial(type) {
  if (!currentMesh) return;
  hideLegend();

  // Clean up dynamic overlays
  if (seamGraphLines) { scene.remove(seamGraphLines); seamGraphLines = null; }
  hideClipControls();
  hideExplodeControls();
  if (pointCloudMode) setPointCloud(false);
  document.getElementById('pointcloud-toggle').classList.remove('active');
  setDualPane(false);
  document.getElementById('dualpane-toggle').classList.remove('active');

  switch (type) {
    case 'default':
      currentMesh.traverse(child => {
        if (!child.isMesh) return;
        if (originalMaterials.has(child.uuid)) {
          child.material = originalMaterials.get(child.uuid).clone();
        }
      });
      break;
    case 'solid':
      currentMesh.traverse(child => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          color: 0x8888aa, roughness: 0.6, metalness: 0.1, wireframe: false,
        });
      });
      break;
    case 'wireframe':
      currentMesh.traverse(child => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          color: 0x6366f1, roughness: 0.5, wireframe: true,
        });
      });
      break;
    case 'checker':
      currentMesh.traverse(child => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          map: checkerTexture, roughness: 0.7, metalness: 0.0, wireframe: false,
        });
      });
      break;
    case 'normal':
      currentMesh.traverse(child => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshNormalMaterial({ wireframe: false });
      });
      break;
    case 'ao':
      currentMesh.traverse(child => {
        if (!child.isMesh) return;
        child.material = new THREE.MeshStandardMaterial({
          color: 0xffffff, roughness: 1.0, metalness: 0.0, wireframe: false, aoMapIntensity: 1.5,
        });
      });
      break;
    case 'distortion':
    case 'ao_heat':
    case 'curvature':
      applyPerFaceColoring(type);
      break;
    case 'chart':
      applyChartColoring();
      break;
    case 'texture':
      if (uploadedTexture) {
        currentMesh.traverse(child => {
          if (!child.isMesh) return;
          child.material = new THREE.MeshStandardMaterial({
            map: uploadedTexture, roughness: 0.5, metalness: 0.0, wireframe: false,
          });
        });
      }
      break;
  }
}

// ─── Screenshot ───
screenshotBtn.addEventListener('click', () => {
  renderer.render(scene, camera);
  canvas.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `mesh_preview_${Date.now()}.png`;
    a.click();
    URL.revokeObjectURL(url);
  });
});

// ─── Reset Camera ───
resetCameraBtn.addEventListener('click', () => {
  if (currentMesh) fitCamera(currentMesh);
});

// ─── Seam Network Graph ───

document.getElementById('seam-graph-toggle').addEventListener('click', async function() {
  if (!currentJobId) return;
  const active = this.classList.toggle('active');

  if (seamGraphLines) {
    scene.remove(seamGraphLines);
    seamGraphLines.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
    seamGraphLines = null;
  }
  if (!active) return;

  const data = await fetchMeshAnalysis();
  if (!data || !data.seam_edges) return;

  seamGraphLines = new THREE.Group();
  const verts = data.vertices;
  const edges = data.seam_edges;

  const edgeMat = new THREE.LineBasicMaterial({ color: 0xef4444, transparent: true, opacity: 0.8, linewidth: 2 });
  const nodeMat = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
  const nodeGeo = new THREE.SphereGeometry(0.015, 6, 6);
  const nodeSet = new Set();

  for (const edge of edges) {
    const [a, b] = edge[0];
    if (a >= verts.length || b >= verts.length) continue;
    const pts = [
      new THREE.Vector3(verts[a][0], verts[a][1], verts[a][2]),
      new THREE.Vector3(verts[b][0], verts[b][1], verts[b][2]),
    ];
    const lineGeo = new THREE.BufferGeometry().setFromPoints(pts);
    seamGraphLines.add(new THREE.LineSegments(lineGeo, edgeMat));

    for (const vi of [a, b]) {
      if (!nodeSet.has(vi)) {
        nodeSet.add(vi);
        const node = new THREE.Mesh(nodeGeo, nodeMat);
        node.position.set(verts[vi][0], verts[vi][1], verts[vi][2]);
        seamGraphLines.add(node);
      }
    }
  }

  scene.add(seamGraphLines);
});

// ─── Clipping Planes ───
let clipPlanes = null;

function showClipControls() {
  document.getElementById('clip-controls').hidden = false;
}
function hideClipControls() {
  document.getElementById('clip-controls').hidden = true;
  setClipping(false);
}

document.getElementById('clip-toggle').addEventListener('click', function() {
  clippingEnabled = this.classList.toggle('active');
  if (clippingEnabled) {
    showClipControls();
    setClipping(true);
  } else {
    hideClipControls();
  }
});

function setClipping(enabled) {
  if (!currentMesh) return;
  if (enabled) {
    const cx = document.getElementById('clip-x').value / 100;
    const cy = document.getElementById('clip-y').value / 100;
    const cz = document.getElementById('clip-z').value / 100;

    currentMesh.traverse(child => {
      if (!child.isMesh) return;
      child.material.clippingPlanes = [
        new THREE.Plane(new THREE.Vector3(-1, 0, 0), cx),
        new THREE.Plane(new THREE.Vector3(0, -1, 0), cy),
        new THREE.Plane(new THREE.Vector3(0, 0, -1), cz),
      ];
      child.material.clipShadows = true;
    });
    renderer.localClippingEnabled = true;
  } else {
    currentMesh.traverse(child => {
      if (!child.isMesh) return;
      child.material.clippingPlanes = [];
    });
    renderer.localClippingEnabled = false;
  }
}

['clip-x', 'clip-y', 'clip-z'].forEach(id => {
  document.getElementById(id).addEventListener('input', () => {
    if (clippingEnabled) setClipping(true);
  });
});

// ─── Exploded View ───

function showExplodeControls() {
  document.getElementById('explode-controls').hidden = false;
}
function hideExplodeControls() {
  document.getElementById('explode-controls').hidden = true;
  setExploded(0);
}

document.getElementById('explode-toggle').addEventListener('click', function() {
  explodeEnabled = this.classList.toggle('active');
  if (explodeEnabled) {
    showExplodeControls();
    const amount = document.getElementById('explode-amount').value / 100;
    setExploded(amount);
  } else {
    hideExplodeControls();
  }
});

document.getElementById('explode-amount').addEventListener('input', function() {
  if (explodeEnabled) setExploded(this.value / 100);
});

function setExploded(amount) {
  if (!currentMesh) return;

  currentMesh.traverse(child => {
    if (!child.isMesh || !child.geometry) return;

    if (!originalPositions.has(child.uuid)) {
      originalPositions.set(child.uuid, child.geometry.attributes.position.array.slice());
    }

    const orig = originalPositions.get(child.uuid);
    const posAttr = child.geometry.attributes.position;
    const box = new THREE.Box3().setFromBufferAttribute(posAttr);
    const center = box.getCenter(new THREE.Vector3());

    for (let i = 0; i < posAttr.count; i++) {
      const ox = orig[i * 3], oy = orig[i * 3 + 1], oz = orig[i * 3 + 2];
      const dir = new THREE.Vector3(ox - center.x, oy - center.y, oz - center.z);
      const len = dir.length();
      if (len > 0.001) dir.normalize();
      posAttr.setXYZ(i, ox + dir.x * amount * len, oy + dir.y * amount * len, oz + dir.z * amount * len);
    }
    posAttr.needsUpdate = true;
  });
}

// ─── Point Cloud Mode ───

function setPointCloud(enabled) {
  pointCloudMode = enabled;
  if (!currentMesh) return;

  if (enabled) {
    // Remove existing point cloud objects
    pointCloudObjects.forEach(obj => { scene.remove(obj); obj.geometry.dispose(); obj.material.dispose(); });
    pointCloudObjects = [];

    currentMesh.traverse(child => {
      if (!child.isMesh || !child.geometry) return;
      if (!originalMeshMaterials.has(child.uuid)) {
        originalMeshMaterials.set(child.uuid, child.material);
      }
      const geo = child.geometry;
      const pointGeo = new THREE.BufferGeometry();
      pointGeo.setAttribute('position', geo.attributes.position.getAttribute ? geo.attributes.position : geo.attributes.position);

      const points = new THREE.Points(pointGeo, new THREE.PointsMaterial({
        color: 0x6366f1,
        size: 0.008,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.8,
      }));
      points.matrixWorld.copy(child.matrixWorld);
      pointCloudObjects.push(points);
      scene.add(points);
    });
  } else {
    pointCloudObjects.forEach(obj => { scene.remove(obj); obj.geometry.dispose(); obj.material.dispose(); });
    pointCloudObjects = [];
  }
}

document.getElementById('pointcloud-toggle').addEventListener('click', function() {
  const active = this.classList.toggle('active');
  setPointCloud(active);
  if (active) {
    applyMaterial('solid');
    document.getElementById('mat-solid').click();
  }
});

// ─── Dual Pane (3D + UV) ───
let uvCanvas = null;
let uvScene = null;
let uvCamera = null;

function setDualPane(enabled) {
  dualPaneEnabled = enabled;
  const vp = document.getElementById('viewport');

  if (enabled) {
    vp.style.display = 'grid';
    vp.style.gridTemplateColumns = '1fr 1fr';
    vp.style.gap = '2px';

    if (!uvCanvas) {
      uvCanvas = document.createElement('canvas');
      uvCanvas.id = 'uv-canvas';
      uvCanvas.style.cssText = 'width:100%;height:100%;display:block;background:#0f1117;';
      vp.appendChild(uvCanvas);

      uvRenderer = new THREE.WebGLRenderer({ canvas: uvCanvas, antialias: true });
      uvRenderer.setPixelRatio(window.devicePixelRatio);
      uvRenderer.setClearColor(0x0f1117);

      uvScene = new THREE.Scene();
      uvCamera = new THREE.OrthographicCamera(-0.1, 1.1, 1.1, -0.1, -1, 10);
      uvCamera.position.set(0.5, 0.5, 5);
      uvCamera.lookAt(0.5, 0.5, 0);

      const uvAmbient = new THREE.AmbientLight(0xffffff, 1.5);
      uvScene.add(uvAmbient);
    }

    renderUVPane();
    onResize();
  } else {
    vp.style.display = '';
    vp.style.gridTemplateColumns = '';
    vp.style.gap = '';
    if (uvCanvas) {
      vp.removeChild(uvCanvas);
      uvCanvas = null;
      uvRenderer = null;
    }
    onResize();
  }
}

function renderUVPane() {
  if (!uvScene || !currentJobId) return;

  // Clear old
  while (uvScene.children.length > 1) uvScene.remove(uvScene.children[uvScene.children.length - 1]);

  fetchMeshAnalysis().then(data => {
    if (!data || !data.vertices || !data.uv_coords || data.uv_coords.length === 0) return;

    const verts = data.vertices;
    const uvs = data.uv_coords;
    const faces = data.faces;

    // Draw UV triangles
    for (const face of faces) {
      if (face.length < 3) continue;
      const pts = face.map(vi => new THREE.Vector3(uvs[vi][0], uvs[vi][1], 0));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const hue = Math.random();
      const color = new THREE.Color().setHSL(hue, 0.6, 0.5);
      const mat = new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide });
      uvScene.add(new THREE.Mesh(geo, mat));

      // Wireframe
      const lineGeo = new THREE.BufferGeometry().setFromPoints([...pts, pts[0]]);
      uvScene.add(new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: 0x444466 })));
    }

    if (uvRenderer) {
      const w = uvCanvas.clientWidth || 400;
      const h = uvCanvas.clientHeight || 400;
      uvRenderer.setSize(w, h);
      uvRenderer.render(uvScene, uvCamera);
    }
  });
}

document.getElementById('dualpane-toggle').addEventListener('click', function() {
  dualPaneEnabled = this.classList.toggle('active');
  setDualPane(dualPaneEnabled);
});

// ─── UV Preview (viewport only) ───
const uvViewportCanvas = document.getElementById('uv-viewport-canvas');
const uvViewportCtx = uvViewportCanvas ? uvViewportCanvas.getContext('2d') : null;
const uvViewportWrap = document.getElementById('uv-viewport-wrap');
let uvPreviewData = null;
let uvPreviewMode = 'wire';
let uvPan = { x: 0, y: 0 };
let uvZoom = 1;
let vpActiveTab = '3d';

const uvChartColors = [
  '#6366f1','#22c55e','#eab308','#ef4444','#a855f7',
  '#06b6d4','#f97316','#ec4899','#14b8a6','#84cc16',
  '#8b5cf6','#f43f5e',
];

function setUVPreviewMode(mode) {
  uvPreviewMode = mode;
  document.querySelectorAll('#uv-viewport-wrap .mat-btn').forEach(b => b.classList.remove('active'));
  const btn = document.getElementById('uv-mode-' + mode);
  if (btn) btn.classList.add('active');
  drawUVViewport();
}

['wire','fill','distortion','chart','seam','coverage'].forEach(mode => {
  const btn = document.getElementById('uv-mode-' + mode);
  if (btn) btn.addEventListener('click', () => setUVPreviewMode(mode));
});

function setupUVPanZoom(canvas, drawFn) {
  let drag = null;
  canvas.addEventListener('mousedown', e => {
    drag = { x: e.clientX - uvPan.x, y: e.clientY - uvPan.y };
  });
  window.addEventListener('mousemove', e => {
    if (!drag) return;
    uvPan.x = e.clientX - drag.x;
    uvPan.y = e.clientY - drag.y;
    drawFn();
  });
  window.addEventListener('mouseup', () => { drag = null; });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    const factor = e.deltaY > 0 ? 0.9 : 1.1;
    uvZoom = Math.max(0.2, Math.min(10, uvZoom * factor));
    drawFn();
  }, { passive: false });
}

if (uvViewportCanvas) setupUVPanZoom(uvViewportCanvas, drawUVViewport);

// Viewport tab toggle
document.getElementById('vp-tab-3d').addEventListener('click', () => {
  vpActiveTab = '3d';
  document.getElementById('vp-tab-3d').classList.add('active');
  document.getElementById('vp-tab-uv').classList.remove('active');
  uvViewportWrap.hidden = true;
  document.getElementById('three-canvas').style.display = '';
  onResize();
});
document.getElementById('vp-tab-uv').addEventListener('click', () => {
  vpActiveTab = 'uv';
  document.getElementById('vp-tab-uv').classList.add('active');
  document.getElementById('vp-tab-3d').classList.remove('active');
  document.getElementById('three-canvas').style.display = 'none';
  uvViewportWrap.hidden = false;
  requestAnimationFrame(() => {
    sizeUVViewportCanvas();
    if (!uvPreviewData || !uvPreviewData.uv_coords || uvPreviewData.uv_coords.length === 0) {
      if (currentJobId) showUVPreview();
    } else {
      drawUVViewport();
    }
  });
});

function sizeUVViewportCanvas() {
  if (!uvViewportCanvas || !uvViewportWrap) return;
  const rect = uvViewportWrap.getBoundingClientRect();
  const dpr = window.devicePixelRatio;
  uvViewportCanvas.width = Math.floor(rect.width * dpr);
  uvViewportCanvas.height = Math.floor(rect.height * dpr);
  uvViewportCanvas.style.width = rect.width + 'px';
  uvViewportCanvas.style.height = rect.height + 'px';
}

function drawUVViewport() {
  if (!uvViewportCtx || !uvViewportDataReady()) return;
  sizeUVViewportCanvas();
  _drawUVToCtx(uvViewportCtx, uvViewportCanvas.width, uvViewportCanvas.height);
}

function uvViewportDataReady() {
  return uvPreviewData && uvPreviewData.uv_coords && uvPreviewData.uv_coords.length > 0;
}

function _drawUVToCtx(ctx, w, h) {
  if (!uvPreviewData || !uvPreviewData.uv_coords.length) return;
  const { minX, minY, rangeX, rangeY, maxRange } = uvPreviewData._bounds;
  const dpr = window.devicePixelRatio;

  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#0f1117';
  ctx.fillRect(0, 0, w, h);

  const margin = 20 * dpr;
  ctx.save();
  ctx.translate(w / 2 + uvPan.x * dpr, h / 2 + uvPan.y * dpr);
  ctx.scale(uvZoom * dpr, uvZoom * dpr);
  ctx.translate(-rangeX / 2 - minX, -rangeY / 2 - minY);

  const uvs = uvPreviewData.uv_coords;
  const faces = uvPreviewData.faces;
  const distortion = uvPreviewData.distortion;
  const chartLabels = uvPreviewData.chart_labels || [];
  const seamEdges = uvPreviewData.seam_edges || [];

  function toScreen(uv) {
    return [uv[0], 1 - uv[1]];
  }

  const facesToDraw = [];
  for (let i = 0; i < faces.length; i++) {
    const face = faces[i];
    if (face.length < 3) continue;
    const pts = face.slice(0, 3).map(vi => toScreen(uvs[vi]));
    facesToDraw.push({ pts, i });
  }

  if (uvPreviewMode === 'fill') {
    for (const { pts, i } of facesToDraw) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      ctx.lineTo(pts[1][0], pts[1][1]);
      ctx.lineTo(pts[2][0], pts[2][1]);
      ctx.closePath();
      const hue = (i * 0.618033988749895) % 1;
      ctx.fillStyle = `hsl(${hue * 360}, 50%, 55%)`;
      ctx.fill();
    }
  } else if (uvPreviewMode === 'distortion') {
    const vals = distortion || [];
    const maxD = Math.max(...vals, 0.001);
    for (const { pts, i } of facesToDraw) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      ctx.lineTo(pts[1][0], pts[1][1]);
      ctx.lineTo(pts[2][0], pts[2][1]);
      ctx.closePath();
      ctx.fillStyle = viridisColormap((vals[i] || 0) / maxD);
      ctx.fill();
    }
  } else if (uvPreviewMode === 'chart') {
    const labels = chartLabels;
    for (const { pts, i } of facesToDraw) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      ctx.lineTo(pts[1][0], pts[1][1]);
      ctx.lineTo(pts[2][0], pts[2][1]);
      ctx.closePath();
      const face = faces[i];
      const chartId = labels[face[0]] || 0;
      const colorIdx = chartId % uvChartColors.length;
      ctx.fillStyle = uvChartColors[colorIdx] + '88';
      ctx.fill();
    }
  } else if (uvPreviewMode === 'coverage') {
    ctx.fillStyle = 'rgba(99,102,241,0.12)';
    ctx.fillRect(minX, minY, rangeX, rangeY);
    for (const { pts } of facesToDraw) {
      ctx.beginPath();
      ctx.moveTo(pts[0][0], pts[0][1]);
      ctx.lineTo(pts[1][0], pts[1][1]);
      ctx.lineTo(pts[2][0], pts[2][1]);
      ctx.closePath();
      ctx.fillStyle = 'rgba(99,102,241,0.25)';
      ctx.fill();
    }
  }

  for (const { pts } of facesToDraw) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    ctx.lineTo(pts[1][0], pts[1][1]);
    ctx.lineTo(pts[2][0], pts[2][1]);
    ctx.closePath();

    if (uvPreviewMode === 'wire' || uvPreviewMode === 'seam' || uvPreviewMode === 'coverage') {
      ctx.strokeStyle = uvPreviewMode === 'seam' ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.2)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    } else {
      ctx.strokeStyle = 'rgba(0,0,0,0.3)';
      ctx.lineWidth = 0.3;
      ctx.stroke();
    }
  }

  if (uvPreviewMode === 'seam') {
    ctx.beginPath();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 2;
    for (const edge of seamEdges) {
      const pair = edge[0];
      if (!pair || pair.length < 2) continue;
      const a = toScreen(uvs[pair[0]]);
      const b = toScreen(uvs[pair[1]]);
      ctx.moveTo(a[0], a[1]);
      ctx.lineTo(b[0], b[1]);
    }
    ctx.stroke();
  }

  ctx.restore();

  ctx.fillStyle = 'rgba(255,255,255,0.3)';
  ctx.font = `${10 * dpr}px monospace`;
  ctx.fillText(`Zoom: ${uvZoom.toFixed(1)}x`, 8 * dpr, h - 8 * dpr);
}

function computeUVMetrics(data) {
  const uvs = data.uv_coords;
  const faces = data.faces;
  const distortion = data.distortion || [];
  const seamEdges = data.seam_edges || [];
  const chartLabels = data.chart_labels || [];

  let numCharts = 0;
  if (chartLabels.length > 0) {
    const chartSet = new Set(chartLabels);
    numCharts = chartSet.size;
  }
  document.getElementById('uv-charts').textContent = numCharts || '-';

  let totalDist = 0, maxDist = 0, distCount = 0;
  for (const d of distortion) {
    if (d > 0) { totalDist += d; distCount++; if (d > maxDist) maxDist = d; }
  }
  document.getElementById('uv-avg-dist').textContent = distCount > 0 ? (totalDist / distCount).toFixed(4) : '-';
  document.getElementById('uv-max-dist').textContent = distCount > 0 ? maxDist.toFixed(4) : '-';

  let seamLen = 0;
  for (const edge of seamEdges) {
    const pair = edge[0];
    if (!pair || pair.length < 2) continue;
    const a = uvs[pair[0]], b = uvs[pair[1]];
    if (a && b) seamLen += Math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2);
  }
  document.getElementById('uv-seam-len').textContent = seamLen > 0 ? seamLen.toFixed(3) : '-';

  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const uv of uvs) {
    if (uv[0] < minX) minX = uv[0];
    if (uv[1] < minY) minY = uv[1];
    if (uv[0] > maxX) maxX = uv[0];
    if (uv[1] > maxY) maxY = uv[1];
  }
  let uvArea = 0;
  for (const face of faces) {
    if (face.length < 3) continue;
    const a = uvs[face[0]], b = uvs[face[1]], c = uvs[face[2]];
    uvArea += Math.abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2;
  }
  document.getElementById('uv-area').textContent = uvArea > 0 ? uvArea.toFixed(4) : '-';

  const bboxArea = (maxX - minX) * (maxY - minY);
  const coverage = bboxArea > 0 ? Math.min(1, uvArea / bboxArea) : 0;
  document.getElementById('uv-coverage').textContent = (coverage * 100).toFixed(1) + '%';
}

// ─── Reset ───
function resetState() {
  currentJobId = null;
  meshAnalysisCache = null;
  uploadedTexture = null;
  originalMaterials.clear();
  document.getElementById('mat-texture-toggle').hidden = true;
  document.getElementById('mat-texture').classList.remove('active');
  uvPreviewData = null;
  document.getElementById('vp-tab-3d').click();
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  meshInfo.hidden = true;
  previewPanel.hidden = true;
  unwrapBtn.disabled = true;
  downloadBtn.hidden = true;
  downloadUvBtn.hidden = true;
  progressPanel.hidden = true;
  resultsPanel.hidden = true;
  viewportEmpty.hidden = false;
  uvOverlay.hidden = true;
  measureOverlay.hidden = true;
  progressFill.style.width = '0%';
  currentMaterial = 'solid';
  Object.values(matBtns).forEach(b => b.classList.remove('active'));
  matBtns.solid.classList.add('active');
  measureMode = false;
  document.getElementById('measure-toggle').classList.remove('active');
  document.getElementById('seam-toggle').classList.remove('active');
  document.getElementById('normals-toggle').classList.remove('active');
  document.getElementById('seam-graph-toggle').classList.remove('active');
  document.getElementById('clip-toggle').classList.remove('active');
  document.getElementById('explode-toggle').classList.remove('active');
  document.getElementById('pointcloud-toggle').classList.remove('active');
  document.getElementById('dualpane-toggle').classList.remove('active');

  hideLegend();
  hideClipControls();
  hideExplodeControls();
  setPointCloud(false);
  setDualPane(false);
  renderer.localClippingEnabled = false;

  clearSeamLines();
  clearMeasurements();
  hideNormals();

  if (seamGraphLines) {
    scene.remove(seamGraphLines);
    seamGraphLines.traverse(c => { if (c.geometry) c.geometry.dispose(); if (c.material) c.material.dispose(); });
    seamGraphLines = null;
  }

  originalPositions.clear();
  originalMeshMaterials.clear();

  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh = null;
  }
}

// ─── Cut Seam Tool ───────────────────────────────────────────────

async function loadCutEdges() {
  if (!currentJobId) return null;
  try {
    const resp = await fetch(`${API}/api/cut-edges/${currentJobId}`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

function toggleCutMode() {
  cutMode = !cutMode;
  joinMode = false;
  const btn = document.getElementById('cut-seam-toggle');
  const panel = document.getElementById('cut-join-panel');
  btn.classList.toggle('active', cutMode);
  document.getElementById('join-island-toggle').classList.remove('active');
  panel.style.display = cutMode || joinMode ? 'block' : 'none';
  document.getElementById('cut-info').style.display = cutMode ? 'block' : 'none';
  document.getElementById('join-info').style.display = 'none';

  if (cutMode) {
    controls.enabled = false;
    loadCutEdges().then(data => {
      cutEdgeData = data;
      selectedCutEdges.clear();
      buildEdgeLines();
      updateCutInfo();
    });
  } else {
    controls.enabled = true;
    clearCutHighlight();
  }
}

function updateCutInfo() {
  const el = document.getElementById('cut-info');
  if (!el) return;
  el.textContent = cutEdgeData
    ? `Click edges to cut (${selectedCutEdges.size} selected / ${cutEdgeData.edges.length} total)`
    : 'Loading edges...';
}

function onViewportClickCut(event) {
  if (!cutMode || !cutEdgeData) return;

  const rect = canvas.getBoundingClientRect();
  cutMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  cutMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  cutRaycaster.setFromCamera(cutMouse, camera);

  // Raycast against edge line segments
  if (cutEdgeLines) {
    const hits = cutRaycaster.intersectObject(cutEdgeLines, true);
    if (hits.length > 0) {
      const idx = hits[0].object.userData.edgeIndex;
      if (idx !== undefined) {
        if (selectedCutEdges.has(idx)) {
          selectedCutEdges.delete(idx);
        } else {
          selectedCutEdges.add(idx);
        }
        highlightCutEdges();
        updateCutInfo();
      }
    }
  }
}

function highlightCutEdges() {
  clearCutHighlight();
  if (!cutEdgeData || selectedCutEdges.size === 0) return;

  const verts = cutEdgeData.vertices;
  const edges = cutEdgeData.edges;
  const positions = [];

  for (const idx of selectedCutEdges) {
    const e = edges[idx];
    positions.push(verts[e[0]][0], verts[e[0]][1], verts[e[0]][2]);
    positions.push(verts[e[1]][0], verts[e[1]][1], verts[e[1]][2]);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.LineBasicMaterial({ color: 0xff4444, linewidth: 3 });
  cutEdgeLines = new THREE.LineSegments(geo, mat);
  cutEdgeLines.userData.isCutHighlight = true;
  scene.add(cutEdgeLines);
}

function clearCutHighlight() {
  if (cutEdgeLines) {
    scene.remove(cutEdgeLines);
    cutEdgeLines.geometry.dispose();
    cutEdgeLines.material.dispose();
    cutEdgeLines = null;
  }
}

function buildEdgeLines() {
  if (!cutEdgeData) return;
  if (cutEdgeLines && cutEdgeLines.userData.isEdgeGraph) {
    scene.remove(cutEdgeLines);
    cutEdgeLines.geometry.dispose();
    cutEdgeLines.material.dispose();
  }

  const verts = cutEdgeData.vertices;
  const edges = cutEdgeData.edges;
  const angles = cutEdgeData.edge_angles || [];
  const positions = [];
  const colors = [];

  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    positions.push(verts[e[0]][0], verts[e[0]][1], verts[e[0]][2]);
    positions.push(verts[e[1]][0], verts[e[1]][1], verts[e[1]][2]);

    // Color by dihedral angle: blue=smooth, red=sharp
    const angle = angles[i] || 0;
    const t = Math.min(angle / 60, 1);
    colors.push(0.2, 0.4, 0.8 - t * 0.6, 0.2, 0.4, 0.8 - t * 0.6);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const mat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.4 });
  cutEdgeLines = new THREE.LineSegments(geo, mat);
  cutEdgeLines.userData.isEdgeGraph = true;
  scene.add(cutEdgeLines);
}

async function applyCut() {
  if (!currentJobId || selectedCutEdges.size === 0) return;

  const method = document.getElementById('cut-method-select').value;
  const form = new FormData();
  form.append('edge_indices', Array.from(selectedCutEdges).join(','));
  form.append('method', method);

  showProgress(`Cutting with ${method}...`);

  try {
    const resp = await fetch(`${API}/api/cut-edges/${currentJobId}`, { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.detail || 'Cut failed');
      hideProgress();
      return;
    }
    const data = await resp.json();

    // Update result
    if (data.uv_coords) {
      uvPreviewData = {
        uv_coords: data.uv_coords,
        faces: data.faces,
        distortion: [],
        chart_labels: [],
        seam_edges: [],
      };
    }

    // Reload mesh preview
    if (data.vertices && data.faces) {
      await loadResultMesh(currentJobId);
    }

    hideProgress();
    cutMode = false;
    selectedCutEdges.clear();
    document.getElementById('cut-seam-toggle').classList.remove('active');
    document.getElementById('cut-join-panel').style.display = 'none';
    controls.enabled = true;

    if (vpActiveTab === 'uv') showUVPreview();
  } catch (e) {
    alert('Cut failed: ' + e.message);
    hideProgress();
  }
}

// ─── Join Islands Tool ───────────────────────────────────────────

async function loadUVIslands() {
  if (!currentJobId) return null;
  try {
    const resp = await fetch(`${API}/api/uv-islands/${currentJobId}`);
    if (!resp.ok) return null;
    return await resp.json();
  } catch { return null; }
}

function toggleJoinMode() {
  joinMode = !joinMode;
  cutMode = false;
  const btn = document.getElementById('join-island-toggle');
  const panel = document.getElementById('cut-join-panel');
  btn.classList.toggle('active', joinMode);
  document.getElementById('cut-seam-toggle').classList.remove('active');
  panel.style.display = cutMode || joinMode ? 'block' : 'none';
  document.getElementById('cut-info').style.display = 'none';
  document.getElementById('join-info').style.display = joinMode ? 'block' : 'none';

  if (joinMode) {
    controls.enabled = false;
    selectedIslands.clear();
    loadUVIslands().then(data => {
      uvIslandData = data;
      updateJoinInfo();
      highlightIslands();
    });
  } else {
    controls.enabled = true;
    clearIslandHighlight();
  }
}

function updateJoinInfo() {
  const el = document.getElementById('join-info');
  if (!el) return;
  el.textContent = uvIslandData
    ? `Click faces to select islands (${selectedIslands.size} selected / ${uvIslandData.num_islands} total)`
    : 'Loading islands...';
}

function highlightIslands() {
  clearIslandHighlight();
  if (!uvIslandData || !uvIslandData.islands) return;

  if (!currentMesh || !currentMesh.geometry) return;

  const posAttr = currentMesh.geometry.getAttribute('position');
  if (!posAttr) return;

  const islands = uvIslandData.islands;
  const palette = [
    [1, 0.4, 0.4], [0.4, 1, 0.4], [0.4, 0.4, 1],
    [1, 1, 0.4], [1, 0.4, 1], [0.4, 1, 1],
    [0.8, 0.6, 0.2], [0.2, 0.8, 0.6], [0.6, 0.2, 0.8],
    [0.9, 0.3, 0.1], [0.1, 0.9, 0.3], [0.3, 0.1, 0.9],
  ];

  const meshFaces = currentMesh.geometry.index
    ? currentMesh.geometry.index.array
    : null;
  const numFaces = meshFaces ? meshFaces.length / 3 : posAttr.count / 3;
  const colors = new Float32Array(posAttr.count * 3);

  for (let fi = 0; fi < numFaces; fi++) {
    let islandIdx = 0;
    for (let ii = 0; ii < islands.length; ii++) {
      if (islands[ii].faces && islands[ii].faces.includes(fi)) {
        islandIdx = ii;
        break;
      }
    }

    const color = palette[islandIdx % palette.length];
    const isSelected = selectedIslands.has(islands[islandIdx]?.id);

    for (let vi = 0; vi < 3; vi++) {
      const vidx = meshFaces ? meshFaces[fi * 3 + vi] : fi * 3 + vi;
      const ci = vidx * 3;
      if (isSelected) {
        colors[ci] = Math.min(color[0] * 1.3, 1);
        colors[ci + 1] = Math.min(color[1] * 1.3, 1);
        colors[ci + 2] = Math.min(color[2] * 1.3, 1);
      } else {
        colors[ci] = color[0] * 0.5;
        colors[ci + 1] = color[1] * 0.5;
        colors[ci + 2] = color[2] * 0.5;
      }
    }
  }

  currentMesh.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  currentMesh.material = new THREE.MeshPhongMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  });
  currentMesh.material.needsUpdate = true;
}

function clearIslandHighlight() {
  if (currentMesh && currentMesh.geometry) {
    currentMesh.geometry.deleteAttribute('color');
    // Restore original material
    const saved = originalMeshMaterials.get(currentMesh);
    if (saved) {
      currentMesh.material = saved;
    } else {
      currentMesh.material = new THREE.MeshPhongMaterial({
        color: 0x8899aa,
        side: THREE.DoubleSide,
      });
    }
  }
  islandFaceMeshes.forEach(m => { scene.remove(m); m.geometry.dispose(); m.material.dispose(); });
  islandFaceMeshes = [];
}

function onViewportClickJoin(event) {
  if (!joinMode || !uvIslandData) return;

  const rect = canvas.getBoundingClientRect();
  cutMouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  cutMouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  cutRaycaster.setFromCamera(cutMouse, camera);

  if (currentMesh) {
    const hits = cutRaycaster.intersectObject(currentMesh, false);
    if (hits.length > 0) {
      const faceIdx = hits[0].faceIndex;
      // Find which island this face belongs to
      if (uvIslandData.islands) {
        for (const island of uvIslandData.islands) {
          if (island.faces && island.faces.includes(faceIdx)) {
            if (selectedIslands.has(island.id)) {
              selectedIslands.delete(island.id);
            } else {
              selectedIslands.add(island.id);
            }
            highlightIslands();
            updateJoinInfo();
            break;
          }
        }
      }
    }
  }
}

async function applyJoin() {
  if (!currentJobId || selectedIslands.size === 0) return;

  const method = document.getElementById('cut-method-select').value;
  const form = new FormData();
  form.append('island_ids', Array.from(selectedIslands).join(','));
  form.append('method', method);

  showProgress(`Joining ${selectedIslands.size} islands with ${method}...`);

  try {
    const resp = await fetch(`${API}/api/join-islands/${currentJobId}`, { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json();
      alert(err.detail || 'Join failed');
      hideProgress();
      return;
    }
    const data = await resp.json();

    if (data.uv_coords) {
      uvPreviewData = {
        uv_coords: data.uv_coords,
        faces: data.faces,
        distortion: [],
        chart_labels: [],
        seam_edges: [],
      };
    }

    if (data.vertices && data.faces) {
      await loadResultMesh(currentJobId);
    }

    hideProgress();
    joinMode = false;
    selectedIslands.clear();
    document.getElementById('join-island-toggle').classList.remove('active');
    document.getElementById('cut-join-panel').style.display = 'none';
    controls.enabled = true;

    if (vpActiveTab === 'uv') showUVPreview();
  } catch (e) {
    alert('Join failed: ' + e.message);
    hideProgress();
  }
}

function showProgress(text) {
  const panel = document.getElementById('progress-panel');
  const bar = document.getElementById('progress-fill');
  const txt = document.getElementById('progress-text');
  if (panel) panel.hidden = false;
  if (bar) bar.style.width = '30%';
  if (txt) txt.textContent = text;
}

function hideProgress() {
  const panel = document.getElementById('progress-panel');
  if (panel) panel.hidden = true;
}

// ─── Cut/Join Event Binding ──────────────────────────────────────

document.getElementById('cut-seam-toggle')?.addEventListener('click', toggleCutMode);
document.getElementById('join-island-toggle')?.addEventListener('click', toggleJoinMode);
document.getElementById('cut-apply-btn')?.addEventListener('click', applyCut);
document.getElementById('cut-clear-btn')?.addEventListener('click', () => {
  selectedCutEdges.clear();
  highlightCutEdges();
  updateCutInfo();
});
document.getElementById('join-apply-btn')?.addEventListener('click', applyJoin);
document.getElementById('join-clear-btn')?.addEventListener('click', () => {
  selectedIslands.clear();
  highlightIslands();
  updateJoinInfo();
});

// Viewport click handler for cut/join
canvas.addEventListener('click', (event) => {
  if (cutMode) onViewportClickCut(event);
  else if (joinMode) onViewportClickJoin(event);
});
