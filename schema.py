from pydantic import BaseModel
from typing import Dict, List

class PredictRequest(BaseModel):
    patient: Dict[str, object]  # allow str for categoricals

class PredictResponse(BaseModel):
    risk: float
    uncertainty: Dict[str, float]
    drivers: List[List]          # [ [feature, shap_value], ... ]
    ood_flag: bool

class WhatIfResponse(BaseModel):
    before: PredictResponse
    after: PredictResponse

class ToggleResponse(PredictResponse):
    """Identical payload to PredictResponse."""

class RecommendCandidate(BaseModel):
    feature: str
    expected_width: float
    expected_reduction: float

class RecommendResponse(BaseModel):
    base_width: float
    ranking: List[RecommendCandidate]

class WhatIfRequest(BaseModel):
    patient: Dict[str, object]
    tweaks: Dict[str, object]

class ToggleRequest(BaseModel):
    patient: Dict[str, object]
    removed_features: List[str]

class RecommendRequest(BaseModel):
    patient: Dict[str, object]
    candidate_features: List[str]
