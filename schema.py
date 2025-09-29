from pydantic import BaseModel
from typing import Dict, List

class PredictRequest(BaseModel):
    patient: Dict[str, object]  # allow str for categoricals

class PredictResponse(BaseModel):
    risk: float
    uncertainty: Dict[str, float]
    drivers: List[List]          # [ [feature, shap_value], ... ]
    ood_flag: bool

class WhatIfRequest(BaseModel):
    patient: Dict[str, object]
    tweaks: Dict[str, object]

class ToggleRequest(BaseModel):
    patient: Dict[str, object]
    removed_features: List[str]

class RecommendRequest(BaseModel):
    patient: Dict[str, object]
    candidate_features: List[str]
