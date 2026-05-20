/* ═══════════════════════════════════════════════════════════════════════════════
   CCTV Sentinel — Client-side Application Logic
   ═══════════════════════════════════════════════════════════════════════════════ */

const API = '';  // Same origin
const POLL_INTERVAL = 1500;  // ms

// ─── State ───────────────────────────────────────────────────────────────────

const state = {
  activeTab: 'detection',
  selectedFiles: [],

  // Detection
  detectRunning: false,
  detectPollTimer: null,

  // Spark
  sparkRunning: false,
  sparkPollTimer: null,

  // Gallery
  alerts: [],
  alertsTotal: 0,
  currentPage: 1,
  pageSize: 24,
  eventFilter: 'all',

  // Logs
  detectLogs: [],
  sparkLogs: [],
};


// ═══════════════════════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupUpload();
  setupControls();
  setupGallery();
  setupLogs();
  checkHealth();
  // Start polling both statuses
  startPolling();
});


// ═══════════════════════════════════════════════════════════════════════════════
//  TABS
// ═══════════════════════════════════════════════════════════════════════════════

function setupTabs() {
  document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      state.activeTab = tab;

      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      btn.classList.add('active');

      document.querySelectorAll('.tab-content').forEach(panel => panel.classList.remove('active'));
      document.getElementById(`${tab}-panel`).classList.add('active');

      // Auto-refresh gallery when switching to it
      if (tab === 'gallery') fetchAlerts();
    });
  });
}


// ═══════════════════════════════════════════════════════════════════════════════
//  UPLOAD
// ═══════════════════════════════════════════════════════════════════════════════

function setupUpload() {
  const fileInput = document.getElementById('file-input');
  const uploadArea = document.getElementById('upload-area');
  const clearBtn = document.getElementById('clear-files-btn');

  fileInput.addEventListener('change', (e) => {
    state.selectedFiles = Array.from(e.target.files);
    renderFileList();
  });

  // Drag and drop
  uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragleave', () => { uploadArea.classList.remove('drag-over'); });
  uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = Array.from(e.dataTransfer.files).filter(f => {
      const ext = f.name.split('.').pop().toLowerCase();
      return ['mp4','avi','mov','mkv','webm','m4v'].includes(ext);
    });
    if (files.length) {
      state.selectedFiles = files;
      renderFileList();
    }
  });

  clearBtn.addEventListener('click', () => {
    state.selectedFiles = [];
    fileInput.value = '';
    renderFileList();
  });
}

function renderFileList() {
  const container = document.getElementById('selected-files');
  const list = document.getElementById('file-list');
  const startBtn = document.getElementById('start-detection-btn');

  if (state.selectedFiles.length === 0) {
    container.style.display = 'none';
    startBtn.disabled = true;
    return;
  }

  container.style.display = 'block';
  startBtn.disabled = state.detectRunning;

  list.innerHTML = state.selectedFiles.map(f => `
    <li class="file-item">
      <span class="file-icon">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"/>
        </svg>
      </span>
      <span class="file-name">${escapeHtml(f.name)}</span>
      <span class="file-size">${formatFileSize(f.size)}</span>
    </li>
  `).join('');
}


// ═══════════════════════════════════════════════════════════════════════════════
//  CONTROLS
// ═══════════════════════════════════════════════════════════════════════════════

function setupControls() {
  document.getElementById('start-spark-btn').addEventListener('click', startSpark);
  document.getElementById('stop-spark-btn').addEventListener('click', stopSpark);
  document.getElementById('start-detection-btn').addEventListener('click', startDetection);
}

async function startSpark() {
  const btn = document.getElementById('start-spark-btn');
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/api/start-spark`, { method: 'POST' });
    const data = await res.json();
    if (!data.success) {
      alert(data.message || 'Failed to start Spark');
      btn.disabled = false;
    }
  } catch (e) {
    alert('Failed to start Spark: ' + e.message);
    btn.disabled = false;
  }
}

async function stopSpark() {
  const btn = document.getElementById('stop-spark-btn');
  btn.disabled = true;
  try {
    const res = await fetch(`${API}/api/stop-spark`, { method: 'POST' });
    const data = await res.json();
    if (!data.success) alert(data.message || 'Failed to stop Spark');
  } catch (e) {
    alert('Failed to stop Spark: ' + e.message);
  }
}

async function startDetection() {
  if (state.selectedFiles.length < 2) {
    alert('Please select at least 2 videos for parallel detection.');
    return;
  }

  const btn = document.getElementById('start-detection-btn');
  btn.disabled = true;

  const formData = new FormData();
  state.selectedFiles.forEach(f => formData.append('videos', f));

  try {
    const res = await fetch(`${API}/api/start-detection`, {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (!data.success) {
      alert(data.message || 'Failed to start detection');
      btn.disabled = false;
    }
  } catch (e) {
    alert('Failed to start detection: ' + e.message);
    btn.disabled = false;
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  POLLING
// ═══════════════════════════════════════════════════════════════════════════════

function startPolling() {
  pollDetection();
  pollSpark();
  setInterval(pollDetection, POLL_INTERVAL);
  setInterval(pollSpark, POLL_INTERVAL);
}

async function pollDetection() {
  try {
    const res = await fetch(`${API}/api/run-status`);
    const data = await res.json();
    if (!data.success) return;

    const s = data.state;
    state.detectRunning = s.running;

    // Update badge
    const badge = document.getElementById('detect-status-badge');
    const dot = badge.querySelector('.status-dot');
    if (s.running) {
      dot.className = 'status-dot running';
      badge.innerHTML = '<span class="status-dot running"></span> Running';
    } else if (s.exitCode !== null && s.exitCode !== undefined) {
      const ok = s.exitCode === 0;
      dot.className = ok ? 'status-dot online' : 'status-dot error';
      badge.innerHTML = `<span class="status-dot ${ok ? 'online' : 'error'}"></span> ${ok ? 'Completed' : 'Error (code ' + s.exitCode + ')'}`;
    } else {
      badge.innerHTML = '<span class="status-dot offline"></span> Idle';
    }

    // Update start button
    const startBtn = document.getElementById('start-detection-btn');
    startBtn.disabled = s.running || state.selectedFiles.length < 2;

    // Progress card
    const progressCard = document.getElementById('progress-card');
    if (s.running || s.finishMs > 0) {
      progressCard.style.display = 'block';
      document.getElementById('progress-videos').textContent = s.videos ? s.videos.length : 0;
      document.getElementById('progress-status').textContent = s.running ? 'Processing...' : 'Finished';
      document.getElementById('progress-exit').textContent = s.exitCode !== null ? s.exitCode : '—';
    }

    // Logs
    updateDetectLogs(s.stdoutTail || [], s.stderrTail || []);

    // Log dot
    const logDot = document.getElementById('detect-log-dot');
    logDot.className = s.running ? 'log-dot active' : 'log-dot';

  } catch (e) { /* silent */ }
}

async function pollSpark() {
  try {
    const res = await fetch(`${API}/api/spark-status`);
    const data = await res.json();
    if (!data.success) return;

    const s = data.state;
    state.sparkRunning = s.running;

    // Update badges
    const badge = document.getElementById('spark-status-badge');
    if (s.running) {
      badge.innerHTML = '<span class="status-dot running"></span> Running';
    } else if (s.exitCode !== null && s.exitCode !== undefined) {
      const ok = s.exitCode === 0;
      badge.innerHTML = `<span class="status-dot ${ok ? 'online' : 'error'}"></span> ${ok ? 'Stopped' : 'Error (code ' + s.exitCode + ')'}`;
    } else {
      badge.innerHTML = '<span class="status-dot offline"></span> Stopped';
    }

    // Buttons
    document.getElementById('start-spark-btn').disabled = s.running;
    document.getElementById('stop-spark-btn').disabled = !s.running;

    // Spark dot on detection page
    const sparkDot = document.getElementById('spark-dot');
    if (sparkDot) sparkDot.style.background = s.running ? '#10b981' : 'rgba(255,255,255,0.5)';

    // Logs
    updateSparkLogs(s.stdoutTail || [], s.stderrTail || []);

    // Log dot
    const logDot = document.getElementById('spark-log-dot');
    logDot.className = s.running ? 'log-dot active' : 'log-dot';

  } catch (e) { /* silent */ }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  LOGS
// ═══════════════════════════════════════════════════════════════════════════════

function setupLogs() {
  document.getElementById('clear-detect-logs').addEventListener('click', () => {
    state.detectLogs = [];
    document.getElementById('detect-log-output').innerHTML = '<p class="log-placeholder">Logs cleared.</p>';
  });
  document.getElementById('clear-spark-logs').addEventListener('click', () => {
    state.sparkLogs = [];
    document.getElementById('spark-log-output').innerHTML = '<p class="log-placeholder">Logs cleared.</p>';
  });
}

function updateDetectLogs(stdout, stderr) {
  const container = document.getElementById('detect-log-output');
  const combined = [
    ...stdout.map(l => ({ text: l, type: 'stdout' })),
    ...stderr.map(l => ({ text: l, type: 'stderr' })),
  ];

  if (combined.length === 0) return;

  const wasAtBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 30;

  container.innerHTML = combined.map(l =>
    `<div class="log-line ${l.type}">${escapeHtml(l.text)}</div>`
  ).join('');

  if (wasAtBottom) container.scrollTop = container.scrollHeight;
}

function updateSparkLogs(stdout, stderr) {
  const container = document.getElementById('spark-log-output');
  const combined = [
    ...stdout.map(l => ({ text: l, type: 'stdout' })),
    ...stderr.map(l => ({ text: l, type: 'stderr' })),
  ];

  if (combined.length === 0) return;

  const wasAtBottom = container.scrollTop + container.clientHeight >= container.scrollHeight - 30;

  container.innerHTML = combined.map(l =>
    `<div class="log-line ${l.type}">${escapeHtml(l.text)}</div>`
  ).join('');

  if (wasAtBottom) container.scrollTop = container.scrollHeight;
}


// ═══════════════════════════════════════════════════════════════════════════════
//  GALLERY
// ═══════════════════════════════════════════════════════════════════════════════

function setupGallery() {
  document.getElementById('refresh-alerts-btn').addEventListener('click', fetchAlerts);

  // Filters
  document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      state.eventFilter = btn.dataset.filter;
      state.currentPage = 1;
      document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      fetchAlerts();
    });
  });

  // Pagination
  document.getElementById('prev-page-btn').addEventListener('click', () => {
    if (state.currentPage > 1) { state.currentPage--; fetchAlerts(); }
  });
  document.getElementById('next-page-btn').addEventListener('click', () => {
    const totalPages = Math.ceil(state.alertsTotal / state.pageSize);
    if (state.currentPage < totalPages) { state.currentPage++; fetchAlerts(); }
  });

  // Lightbox close
  document.getElementById('lightbox-close').addEventListener('click', closeLightbox);
  document.getElementById('lightbox-overlay').addEventListener('click', (e) => {
    if (e.target === document.getElementById('lightbox-overlay')) closeLightbox();
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });
}

async function fetchAlerts() {
  const offset = (state.currentPage - 1) * state.pageSize;
  let url = `${API}/api/alerts?limit=${state.pageSize}&offset=${offset}`;
  if (state.eventFilter !== 'all') url += `&event_type=${state.eventFilter}`;

  try {
    const res = await fetch(url);
    const data = await res.json();

    if (!data.success) {
      console.error('Failed to fetch alerts:', data.error);
      return;
    }

    state.alerts = data.alerts || [];
    state.alertsTotal = data.total || 0;
    renderGallery();
  } catch (e) {
    console.error('Error fetching alerts:', e);
  }
}

function renderGallery() {
  const grid = document.getElementById('gallery-grid');
  const countEl = document.getElementById('alert-count');
  const pagination = document.getElementById('pagination');
  const emptyEl = document.getElementById('gallery-empty');

  countEl.textContent = `${state.alertsTotal} alert${state.alertsTotal !== 1 ? 's' : ''}`;

  if (state.alerts.length === 0) {
    grid.innerHTML = '';
    grid.appendChild(emptyEl);
    emptyEl.style.display = '';
    pagination.style.display = 'none';
    return;
  }

  // Remove empty state
  if (emptyEl && emptyEl.parentNode === grid) grid.removeChild(emptyEl);

  grid.innerHTML = state.alerts.map(alert => {
    const imgSrc = alert.image_path
      ? `${API}/api/hdfs-image?path=${encodeURIComponent(alert.image_path)}`
      : '';
    const eventType = (alert.event_type || 'unknown').toLowerCase();
    const confidence = alert.confidence != null ? (alert.confidence * 100).toFixed(1) + '%' : '—';
    const timestamp = alert.timestamp ? formatTimestamp(alert.timestamp) : '—';
    const cameraId = alert.camera_id || '—';
    const alertId = alert.id || '—';

    return `
      <div class="alert-card" data-alert-id="${alertId}" onclick='openLightbox(${JSON.stringify(alert)})'>
        <img
          class="alert-card-image loading"
          data-src="${imgSrc}"
          alt="${eventType} detection"
          onload="this.classList.remove('loading')"
          onerror="this.classList.remove('loading'); this.style.opacity='0.3';"
        />
        <div class="alert-card-body">
          <div class="alert-card-top">
            <span class="event-badge ${eventType}">${eventType === 'fire' ? '🔥' : '🔫'} ${eventType}</span>
            <span class="confidence-value">${confidence}</span>
          </div>
          <div class="alert-card-meta">
            <div class="meta-item">
              <span class="meta-label">ID</span>
              <span class="meta-value">#${alertId}</span>
            </div>
            <div class="meta-item">
              <span class="meta-label">Camera</span>
              <span class="meta-value">${escapeHtml(cameraId)}</span>
            </div>
            <div class="meta-item" style="grid-column: 1 / -1;">
              <span class="meta-label">Timestamp</span>
              <span class="meta-value">${timestamp}</span>
            </div>
          </div>
        </div>
      </div>
    `;
  }).join('');

  // Lazy load images
  grid.querySelectorAll('img[data-src]').forEach(img => {
    const src = img.dataset.src;
    if (src) img.src = src;
  });

  // Pagination
  const totalPages = Math.ceil(state.alertsTotal / state.pageSize);
  if (totalPages > 1) {
    pagination.style.display = 'flex';
    document.getElementById('prev-page-btn').disabled = state.currentPage <= 1;
    document.getElementById('next-page-btn').disabled = state.currentPage >= totalPages;
    document.getElementById('page-info').textContent = `Page ${state.currentPage} of ${totalPages}`;
  } else {
    pagination.style.display = 'none';
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  LIGHTBOX
// ═══════════════════════════════════════════════════════════════════════════════

function openLightbox(alert) {
  const overlay = document.getElementById('lightbox-overlay');
  const image = document.getElementById('lightbox-image');
  const info = document.getElementById('lightbox-info');

  const imgSrc = alert.image_path
    ? `${API}/api/hdfs-image?path=${encodeURIComponent(alert.image_path)}`
    : '';

  image.src = imgSrc;

  const eventType = (alert.event_type || 'unknown').toLowerCase();
  const confidence = alert.confidence != null ? (alert.confidence * 100).toFixed(2) + '%' : '—';
  const timestamp = alert.timestamp ? formatTimestamp(alert.timestamp) : '—';

  info.innerHTML = `
    <div class="modal-info-item">
      <span class="modal-info-label">Alert ID</span>
      <span class="modal-info-value">#${alert.id || '—'}</span>
    </div>
    <div class="modal-info-item">
      <span class="modal-info-label">Event Type</span>
      <span class="modal-info-value" style="color: ${eventType === 'fire' ? '#f97316' : '#ef4444'}">
        ${eventType === 'fire' ? '🔥' : '🔫'} ${eventType.toUpperCase()}
      </span>
    </div>
    <div class="modal-info-item">
      <span class="modal-info-label">Confidence</span>
      <span class="modal-info-value">${confidence}</span>
    </div>
    <div class="modal-info-item">
      <span class="modal-info-label">Camera ID</span>
      <span class="modal-info-value">${escapeHtml(alert.camera_id || '—')}</span>
    </div>
    <div class="modal-info-item">
      <span class="modal-info-label">Timestamp</span>
      <span class="modal-info-value">${timestamp}</span>
    </div>
    <div class="modal-info-item">
      <span class="modal-info-label">HDFS Path</span>
      <span class="modal-info-value" style="font-size:0.75rem; word-break:break-all;">${escapeHtml(alert.image_path || '—')}</span>
    </div>
  `;

  overlay.style.display = 'flex';
  document.body.style.overflow = 'hidden';
}

function closeLightbox() {
  document.getElementById('lightbox-overlay').style.display = 'none';
  document.getElementById('lightbox-image').src = '';
  document.body.style.overflow = '';
}


// ═══════════════════════════════════════════════════════════════════════════════
//  HEALTH CHECK
// ═══════════════════════════════════════════════════════════════════════════════

async function checkHealth() {
  const dot = document.getElementById('backend-dot');
  const label = document.getElementById('backend-label');
  try {
    const res = await fetch(`${API}/api/health`);
    const data = await res.json();
    if (data.success) {
      dot.className = 'status-dot online';
      label.textContent = 'Online';
    } else {
      dot.className = 'status-dot error';
      label.textContent = 'Error';
    }
  } catch (e) {
    dot.className = 'status-dot error';
    label.textContent = 'Offline';
  }
}


// ═══════════════════════════════════════════════════════════════════════════════
//  UTILITIES
// ═══════════════════════════════════════════════════════════════════════════════

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
}

function formatTimestamp(ms) {
  try {
    const d = new Date(ms);
    return d.toLocaleString('en-IN', {
      year: 'numeric', month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit', second: '2-digit',
      hour12: true,
    });
  } catch (e) {
    return String(ms);
  }
}
