"""App creation."""
from fastapi import FastAPI

from src.infrastructure.entrypoint import fraud_score as fraud_score_entrypoint
from src.infrastructure.entrypoint import health


def include_routers(app: FastAPI) -> None:
    """Include all routers in the FastAPI app."""
    app.include_router(health.router)
    app.include_router(fraud_score_entrypoint.router)


def create_app() -> FastAPI:

    app = FastAPI(
        title="Rinha 2026",
        description="Rinha 2026",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    include_routers(app)

    return app


main_app = create_app()
