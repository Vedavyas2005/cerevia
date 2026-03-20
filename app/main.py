# app/main.py

import sys
import io
import base64
from pathlib import Path
from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHECKPOINT_DIR, SERVE_MODEL, IMAGE_SIZE,
    CLASS_NAMES, APP_HOST, APP_PORT,
)
from models                 import get_model
from explainability.gradcam import GradCAM

# ─────────────────────────────────────────────────────────────────────────────
# Global model state
# ─────────────────────────────────────────────────────────────────────────────

DEVICE: torch.device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL:  torch.nn.Module | None = None
CAM:    GradCAM | None         = None


def _load_model():
    global MODEL, CAM
    ckpt_path = CHECKPOINT_DIR / f"{SERVE_MODEL}_best.pth"
    if not ckpt_path.exists():
        print(f"WARNING: checkpoint not found at {ckpt_path}.")
        return
    MODEL = get_model(SERVE_MODEL).to(DEVICE)
    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(ckpt["state_dict"])
    MODEL.eval()
    CAM = GradCAM(MODEL)
    print(f"Model loaded: {SERVE_MODEL}  |  device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app      = FastAPI(title="Alzheimer's MRI Classifier", version="1.0", lifespan=lifespan)
BASE_DIR = Path(__file__).parent

# ── CORS — allows Netlify frontend to call this API ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your Netlify URL after deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def preprocess(pil_image: Image.Image) -> torch.Tensor:
    return _transform(pil_image.convert("RGB")).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    """Ping endpoint — keeps HuggingFace Space awake."""
    return {"status": "ok", "model": SERVE_MODEL, "device": str(DEVICE)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        contents  = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    tensor = preprocess(pil_image)

    try:
        heatmap, pred_idx, confidence = CAM(tensor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    with torch.no_grad():
        logits = MODEL(tensor)
        probs  = F.softmax(logits, dim=-1)[0].cpu().tolist()

    probabilities = {cls: round(p, 4) for cls, p in zip(CLASS_NAMES, probs)}

    overlay_pil = GradCAM.overlay(pil_image, heatmap)
    buf = io.BytesIO()
    overlay_pil.save(buf, format="PNG")
    gradcam_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return JSONResponse({
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence":      round(confidence, 4),
        "probabilities":   probabilities,
        "gradcam_image":   gradcam_b64,
    })


@app.get("/models")
async def list_models():
    available = [
        p.stem.replace("_best", "")
        for p in CHECKPOINT_DIR.glob("*_best.pth")
    ]
    return {"models": available, "serving": SERVE_MODEL}


if __name__ == "__main__":
    import uvicorn
    # Use 7860 for HuggingFace, 8000 for local
    port = 7860 if Path("/.dockerenv").exists() else APP_PORT
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)