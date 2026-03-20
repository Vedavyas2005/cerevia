# Cerevia — Alzheimer's MRI Classifier

An explainable deep learning system for Alzheimer's stage detection from brain MRI scans.
Benchmarks three architectures trained from scratch and deploys the best as a clinical decision-support tool.

## Live Demo
🌐 Frontend: [cerevia.netlify.app](https://cerevia.netlify.app)
🔗 Backend API: [ved2005-alzheimers-mri-backend.hf.space](https://ved2005-alzheimers-mri-backend.hf.space)
📖 API Docs: [/docs](https://ved2005-alzheimers-mri-backend.hf.space/docs)

## What it does
- Classifies brain MRI into 4 Alzheimer's stages: Non Demented, Very Mild, Mild, Moderate
- Generates Grad-CAM heatmaps showing which brain regions drove the prediction
- Trained on 80,000 OASIS neuroimaging scans

## Results

| Model | Accuracy | F1 Macro | AUC |
|---|---|---|---|
| ResNet-18 | 99.76% | 99.71% | 1.0000 |
| EfficientNet-B0 | **99.97%** | **99.97%** | **1.0000** |
| ViT-Tiny | 67.07% | 44.22% | 0.8744 |

EfficientNet-B0 selected for deployment. ViT underperformance confirms published findings
that transformers require pretraining to match CNNs on small medical imaging datasets.

## Stack
- PyTorch (from scratch, no pretrained weights)
- FastAPI + Uvicorn
- Grad-CAM explainability
- Deployed: HuggingFace Spaces (backend) + Netlify (frontend)

## Run locally
\`\`\`bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python app/main.py
\`\`\`

## Dataset
OASIS Neuroimaging Dataset — ninadaithal/imagesoasis on Kaggle

⚠️ For research purposes only. Not a medical device.