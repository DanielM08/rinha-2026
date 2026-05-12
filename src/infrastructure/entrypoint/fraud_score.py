from fastapi import APIRouter

from src.domain.fraud_score import FraudScoreRequest, FraudScoreResponse

router = APIRouter()


@router.post("/fraud-score")
def fraud_score(body: FraudScoreRequest) -> FraudScoreResponse:
    return FraudScoreResponse(approved=False, fraud_score=1.0)
