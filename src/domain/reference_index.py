import gzip
import json
import logging
import os
import time
from pathlib import Path

import faiss
import numpy as np

_TOTAL_DIMENSIONS = 14
_PQ_SUBQUANTIZERS = 7  # must divide _TOTAL_DIMENSIONS; 7×2-dim sub-quantizers → 8× compression
_logger = logging.getLogger(__name__)


class ReferenceIndex:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        nprobe = int(os.getenv("FAISS_NPROBE", "16"))

        faiss_path, labels_path = _binary_paths(path)
        t0 = time.perf_counter()

        if faiss_path.exists() and labels_path.exists():
            # Fast path (Docker): load the pre-built binary files produced by
            # scripts/build_index.py at image build time. No JSON parsing,
            # no training — fits comfortably within the 160 MB container limit.
            _logger.info("Loading pre-built index from %s", faiss_path)
            self._index = faiss.read_index(str(faiss_path))
            raw_labels = np.load(str(labels_path))
            self._labels = ["fraud" if v else "legit" for v in raw_labels]
        else:
            # Fallback (local dev): build the index from the JSON file directly.
            _logger.info("No pre-built index found — building from %s", path)
            self._index, self._labels = _build_from_json(path)

        self._index.nprobe = nprobe

        elapsed = time.perf_counter() - t0
        fraud_count = self._labels.count("fraud")
        _logger.info(
            "Reference index ready: %d vectors (%d fraud, %d legit) — nprobe=%d in %.2fs",
            len(self._labels),
            fraud_count,
            len(self._labels) - fraud_count,
            nprobe,
            elapsed,
        )

    def search(self, vector: list[float], k: int = 5) -> list[str]:
        query = np.array([vector], dtype=np.float32)
        _, indices = self._index.search(query, k)
        return [self._labels[i] for i in indices[0]]


def _binary_paths(path: Path) -> tuple[Path, Path]:
    """Derive .faiss and _labels.npy paths from any reference file path.

    Examples:
        resources/references.json.gz  →  resources/references.faiss
                                         resources/references_labels.npy
        resources/example-references.json  →  resources/example-references.faiss
                                               resources/example-references_labels.npy
    """
    stem = path.name.split(".")[0]  # strip all suffixes (.json, .gz, etc.)
    base = path.parent / stem
    return base.with_suffix(".faiss"), Path(str(base) + "_labels.npy")


def _build_from_json(path: Path) -> tuple[faiss.Index, list[str]]:
    """Build an IndexIVFPQ from a JSON/.json.gz file (local dev fallback)."""
    nlist = int(os.getenv("FAISS_NLIST", "1024"))

    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            records = json.load(f)
    else:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)

    vectors = np.array([r["vector"] for r in records], dtype=np.float32)
    labels: list[str] = [r["label"] for r in records]

    _logger.info("Training IndexIVFPQ (nlist=%d, m=%d) on %d vectors…", nlist, _PQ_SUBQUANTIZERS, len(vectors))
    quantizer = faiss.IndexFlatL2(_TOTAL_DIMENSIONS)
    index = faiss.IndexIVFPQ(quantizer, _TOTAL_DIMENSIONS, nlist, _PQ_SUBQUANTIZERS, 8)
    index.train(vectors)
    index.add(vectors)
    return index, labels
