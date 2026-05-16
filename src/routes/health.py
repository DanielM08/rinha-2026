from fastapi import APIRouter
from fastapi.responses import Response

router = APIRouter()


@router.get("/ready")
def ready() -> Response:
    return Response(status_code=200)
