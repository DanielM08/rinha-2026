"""App creation."""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

from src.domain.reference_index import ReferenceIndex
from src.routes import fraud_score as fraud_score_entrypoint
from src.routes import health

_DEFAULT_REFERENCES_PATH = "resources/references.json.gz"


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("REFERENCES_PATH", _DEFAULT_REFERENCES_PATH)
    app.state.reference_index = ReferenceIndex(path)
    yield


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
        lifespan=lifespan,
    )

    include_routers(app)

    return app


main_app = create_app()
