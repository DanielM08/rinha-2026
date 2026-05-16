import logging

from fastapi import APIRouter, Request

from src.domain.fraud_score import FraudScoreRequest, FraudScoreResponse
from src.domain.vectorizer import FraudVectorizer

router = APIRouter()
_logger = logging.getLogger(__name__)
_vectorizer = FraudVectorizer()


@router.post("/fraud-score")
def fraud_score(body: FraudScoreRequest, request: Request) -> FraudScoreResponse:
    vector = _vectorizer.vectorize(body)
    labels = request.app.state.reference_index.search(vector, k=5)
    score = labels.count("fraud") / len(labels)
    approved = score < 0.6
    _logger.info(
        "tx=%s score=%.2f approved=%s neighbors=%s",
        body.id,
        score,
        approved,
        labels,
    )
    return FraudScoreResponse(approved=approved, fraud_score=score)
