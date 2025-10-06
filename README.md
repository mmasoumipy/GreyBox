This project packages a small end-to-end workflow for training an
ensemble diabetes risk model, serving calibrated predictions with
uncertainty estimates, and providing an interactive Streamlit UI for
what-if analysis.

## 1. Train the model ensemble

```bash
pip install -r requirements.txt
python train.py
```

Running the training script will fit a LightGBM ensemble, calibrate it
with isotonic regression, estimate conformal prediction intervals,
train an isolation forest for out-of-distribution detection, and store
all artifacts in `./artifacts`.

## 2. Launch the Streamlit app

```bash
streamlit run streamlit_app.py
```

The UI lets you upload a dataset (or use a synthetic fallback), inspect
predictions for a selected patient, view conformal uncertainty bands,
see SHAP local feature attributions, and explore what-if scenarios for
individual features. You can also simulate dropping features and obtain
expected reductions in uncertainty from re-running tests.

## 3. Serve the API

```bash
uvicorn api:app --reload
```

The FastAPI service exposes:

- `POST /predict` – return risk, conformal bounds, epistemic proxy, and
  top SHAP drivers for a patient.
- `POST /what-if` – compare predictions before/after tweaking feature
  values.
- `POST /toggle` – simulate removing information by resetting selected
  features.
- `POST /recommend` – rank numeric labs/tests by expected reduction of
  the conformal band width.
- `GET /meta` – surface feature names, ranges, and units for building
  custom clients.

All endpoints accept/return JSON following the Pydantic models defined
in `schema.py`.