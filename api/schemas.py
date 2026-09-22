from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    target_accuracy_met: bool
    target_qwk_met: bool


class QualityMetrics(BaseModel):
    accepted: bool
    reasons: List[str]
    brightness: Optional[float] = None
    contrast: Optional[float] = None
    blur_score: Optional[float] = None
    retina_circularity: Optional[float] = None


class ReferableDRDetails(BaseModel):
    is_referable: bool
    probability: float
    threshold: float


class PredictionDetails(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    expected_continuous_score: float
    probabilities: Dict[str, float]
    referable_dr: ReferableDRDetails
    decision_rule: str


class ClinicalRecommendation(BaseModel):
    severity: str
    color: str
    referable: bool
    urgency: str
    action: str


class PredictionResponse(BaseModel):
    accepted: bool
    quality: Dict[str, Any]
    prediction: Optional[PredictionDetails] = None
    clinical_recommendation: Optional[ClinicalRecommendation] = None
    gradcam: Optional[Dict[str, Any]] = None


class ExplainResponse(BaseModel):
    accepted: bool
    predicted_class: int
    class_name: str
    gradcam_base64: str


class ExperimentsResponse(BaseModel):
    final_performance: Dict[str, Any]
    optimal_thresholds: List[float]
    stages: List[Dict[str, Any]]
