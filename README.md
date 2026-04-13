# Hospitality AI Project RoboRoarZ 2026

## Quickstart (local)

### 1) Start backend

Make sure you are into the workspace and simply run the bash scripts

```bash
mkdir hospitality-ai
git clone https://github.com/patuan07/HospitalityAI.git
cd HospitalityAI
bash scripts/run_backend.sh 
```

or do the following manually

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open health check:

- `GET http://localhost:8000/health`

### 2) Start Streamlit UI

Simply run the bash scripts

```bash
cd hospitality-ai-product-template_final
bash scripts/run_webapp.sh 
```

or run the following codes manually

```bash
cd webapp_streamlit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8501
```

Open UI:

- `http://localhost:8501`

---

## API Contract (do not break)

The response schema is defined in:

- `backend/app/schemas.py`

and summarized in:

- `shared/contracts.md`

If you change internal model code, **keep the output keys stable** so your UI and judging scripts don't break.

---

## Where to plug in your ML stages

Edit or replace these files:

- `backend/ml/stage1_binary.py` (Stage 1)
- `backend/ml/stage2_classifier.py` (Stage 2)
- `backend/ml/stage3_weak_cam.py` (Stage 3)
- `backend/ml/stage4_geometry.py` (Stage 4)
- `backend/ml/stage5_robustness.py` (Stage 5, optional)


The orchestration is done in:

- `backend/ml/pipeline.py`

---

## Notes on dependencies

This template runs **without** heavy ML libraries by default.

If your team uses torch / Grad-CAM:

- add your dependencies to `backend/requirements.txt`
- update `stage1_binary.py`, `stage2_classifier.py`, and `stage3_weak_cam.py`

---

## Docker (optional - This is not necessary and will not give you any advantage in the competition)

Advanced teams can run:

```bash
cd docker
docker compose up --build
```

Backend: `http://localhost:8000`
UI: `http://localhost:8501`
# HospitalityAISTreamLit
