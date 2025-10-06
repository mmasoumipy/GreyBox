"""FastAPI service exposing prediction + uncertainty utilities."""
from typing import Dict

from fastapi import FastAPI, HTTPException

from model_utils import (
    get_meta,
    predict_with_uncertainty,
    recommend_tests,
    toggle_features,
    what_if,
)
from schema import (
    PredictRequest,
    PredictResponse,
    RecommendRequest,
    RecommendResponse,
    ToggleRequest,
    ToggleResponse,
    WhatIfRequest,
    WhatIfResponse,
)

app = FastAPI(
    title="Diabetes Risk + Uncertainty API",
    description=(
        "Serve calibrated risk predictions, conformal uncertainty bands, "
        "and explainability utilities for the diabetes model ensemble."
    ),
    version="0.1.0",
)


@app.get("/meta")
def read_meta() -> Dict:
    """Return metadata describing features, ranges, and units."""
    return get_meta()


def _as_response(payload: Dict) -> PredictResponse:
    """Convert raw dictionary output into the response schema."""
    payload = payload.copy()
    payload["drivers"] = [[str(f), float(v)] for f, v in payload.get("drivers", [])]
    return PredictResponse(**payload)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        result = predict_with_uncertainty(req.patient)
    except ValueError as exc:  # guard against casting issues
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _as_response(result)


@app.post("/what-if", response_model=WhatIfResponse)
def run_what_if(req: WhatIfRequest) -> WhatIfResponse:
    try:
        res = what_if(req.patient, req.tweaks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WhatIfResponse(before=_as_response(res["before"]), after=_as_response(res["after"]))


@app.post("/toggle", response_model=ToggleResponse)
def toggle(req: ToggleRequest) -> ToggleResponse:
    try:
        res = toggle_features(req.patient, req.removed_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _as_response(res)


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    try:
        res = recommend_tests(req.patient, req.candidate_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RecommendResponse(**res)