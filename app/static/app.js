// app.js

const fileInput   = document.getElementById("fileInput");
const dropZone    = document.getElementById("dropZone");
const previewWrap = document.getElementById("previewWrap");
const previewImg  = document.getElementById("preview");
const analyseBtn  = document.getElementById("analyseBtn");
const resultsCard = document.getElementById("resultsCard");
const loadingOv   = document.getElementById("loadingOverlay");

let selectedFile = null;

// ── File selection ────────────────────────────────────────────────────────────

fileInput.addEventListener("change", e => setFile(e.target.files[0]));

dropZone.addEventListener("dragover",  e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) setFile(file);
});
dropZone.addEventListener("click", () => fileInput.click());

function setFile(file) {
  if (!file) return;
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  document.getElementById("originalImg").src = url;
  previewWrap.style.display = "block";
  dropZone.style.display    = "none";
  analyseBtn.disabled       = false;
  resultsCard.style.display = "none";
}

function clearAll() {
  selectedFile              = null;
  fileInput.value           = "";
  previewImg.src            = "";
  previewWrap.style.display = "none";
  dropZone.style.display    = "block";
  analyseBtn.disabled       = true;
  resultsCard.style.display = "none";
}

// ── Analyse ───────────────────────────────────────────────────────────────────

async function analyse() {
  if (!selectedFile) return;

  loadingOv.style.display = "flex";
  analyseBtn.disabled     = true;

  const form = new FormData();
  form.append("file", selectedFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();

    if (!res.ok) {
      alert("Error: " + (data.detail || "Unknown error"));
      return;
    }

    renderResults(data);

  } catch (err) {
    alert("Network error: " + err.message);
  } finally {
    loadingOv.style.display = "none";
    analyseBtn.disabled     = false;
  }
}

// ── Render results ────────────────────────────────────────────────────────────

function renderResults(data) {
  document.getElementById("resultTag").textContent  = data.predicted_class;
  document.getElementById("resultConf").textContent =
    `Confidence: ${(data.confidence * 100).toFixed(1)}%`;

  // Probability bars
  const container = document.getElementById("probBars");
  container.innerHTML = "";

  const sorted = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1]);

  sorted.forEach(([cls, prob], idx) => {
    const pct  = (prob * 100).toFixed(1);
    const isTop = idx === 0;
    container.innerHTML += `
      <div class="prob-row">
        <div class="prob-label">
          <span>${cls}</span>
          <span>${pct}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill ${isTop ? "top" : ""}"
               style="width: ${pct}%"></div>
        </div>
      </div>`;
  });

  // Grad-CAM
  document.getElementById("gradcamImg").src =
    "data:image/png;base64," + data.gradcam_image;

  resultsCard.style.display = "block";
  resultsCard.scrollIntoView({ behavior: "smooth" });
}
