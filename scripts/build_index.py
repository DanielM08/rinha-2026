"""
Runs at `docker build` time to convert references.json.gz into compact binary
files loaded instantly at container startup.

Streaming strategy (no full file loaded into memory at once):

  1. Stream the first TRAIN_SAMPLE vectors into a small numpy array → index.train()
  2. Continue streaming the rest in batches of ADD_BATCH_SIZE → index.add(batch)
  3. Accumulate labels as a uint8 numpy array grown in chunks

Peak RAM: max(TRAIN_SAMPLE, ADD_BATCH_SIZE) × 14 × 4 B ≈ a few MB.

Output files:
  resources/references.faiss        — ~22 MB compressed Faiss index
  resources/references_labels.npy   — ~3 MB uint8 label array (0=legit, 1=fraud)
"""

import gzip
import sys
from pathlib import Path

import faiss
import ijson
import numpy as np

_DIMS = 14
_PQ_SUBQUANTIZERS = 7  # must divide _DIMS
_NLIST = 1024

# Number of vectors used to train IVF+PQ centroids.
# Faiss recommends ≥ 39 × nlist; 64 × nlist gives comfortable headroom.
_TRAIN_SAMPLE = _NLIST * 64  # 65 536 vectors, ~3.5 MB float32

_ADD_BATCH = 50_000  # vectors flushed to the index per iteration

RESOURCES = Path("resources")
SRC = RESOURCES / "references.json.gz"
DST_INDEX = RESOURCES / "references.faiss"
DST_LABELS = RESOURCES / "references_labels.npy"


def _iter_records(path: Path):
    """Yield (vector, label) tuples by streaming the JSON without loading it all."""
    with gzip.open(path, "rb") as f:
        for record in ijson.items(f, "item"):
            yield record["vector"], record["label"]


def main() -> None:
    print(f"[build_index] Streaming {SRC} …", flush=True)

    # ── Phase 1: collect training sample ──────────────────────────────────
    train_vecs: list[list[float]] = []
    batch_vecs: list[list[float]] = []
    all_labels: list[int] = []
    trained = False
    total = 0

    quantizer = faiss.IndexFlatL2(_DIMS)
    index = faiss.IndexIVFPQ(quantizer, _DIMS, _NLIST, _PQ_SUBQUANTIZERS, 8)

    for vector, label in _iter_records(SRC):
        label_int = 1 if label == "fraud" else 0
        all_labels.append(label_int)
        total += 1

        if not trained:
            train_vecs.append(vector)
            if len(train_vecs) >= _TRAIN_SAMPLE:
                arr = np.array(train_vecs, dtype=np.float32)
                print(
                    f"[build_index] Training on {len(train_vecs)} vectors …", flush=True
                )
                index.train(arr)
                # add the training vectors as the first batch
                index.add(arr)
                del train_vecs, arr
                trained = True
        else:
            batch_vecs.append(vector)
            if len(batch_vecs) >= _ADD_BATCH:
                index.add(np.array(batch_vecs, dtype=np.float32))
                batch_vecs = []

        if total % 500_000 == 0:
            print(f"[build_index]   … {total:,} vectors processed", flush=True)

    # flush remaining batch
    if not trained:
        # dataset smaller than TRAIN_SAMPLE (e.g. test run on a small file)
        arr = np.array(train_vecs, dtype=np.float32)
        index.train(arr)
        index.add(arr)
    elif batch_vecs:
        index.add(np.array(batch_vecs, dtype=np.float32))

    faiss.write_index(index, str(DST_INDEX))
    labels_arr = np.array(all_labels, dtype=np.uint8)
    np.save(str(DST_LABELS), labels_arr)

    fraud_count = int(labels_arr.sum())
    print(
        f"[build_index] Done. {total:,} vectors ({fraud_count:,} fraud, "
        f"{total - fraud_count:,} legit). Wrote {DST_INDEX} and {DST_LABELS}.",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
