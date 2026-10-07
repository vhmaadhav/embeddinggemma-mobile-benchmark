"""Small accuracy suite: STS-B (Spearman) and a SciFact subset (NDCG@10)."""

import json

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from scipy.stats import spearmanr

from .config import (
    DATA,
    DOC_TEMPLATE,
    MRL_DIMS,
    QUERY_PREFIX,
    RETRIEVAL_CORPUS_SIZE,
    SEED,
    STS_PREFIX,
)
from .encoder import Encoder, truncate


def _fetch(repo: str, path: str) -> str:
    return hf_hub_download(repo, path, repo_type="dataset", cache_dir=DATA)


def _jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf8") as f:
        return [json.loads(line) for line in f]


def load_stsb(split: str = "test") -> pd.DataFrame:
    """Columns: sentence1, sentence2, score (0..1)."""
    return pd.read_parquet(_fetch("sentence-transformers/stsb", f"data/{split}-00000-of-00001.parquet"))


def load_scifact() -> tuple[dict, dict, dict]:
    """Test queries, a fixed corpus subset, and qrels {qid: {doc_id: rel}}."""
    qrels: dict[str, dict[str, int]] = {}
    for r in _jsonl(_fetch("mteb/scifact", "qrels/test.jsonl")):
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r["score"])
    queries = {
        str(q["_id"]): q["text"]
        for q in _jsonl(_fetch("mteb/scifact", "queries.jsonl"))
        if str(q["_id"]) in qrels
    }
    docs = {str(d["_id"]): d for d in _jsonl(_fetch("mteb/scifact", "corpus.jsonl"))}

    relevant = sorted({d for rels in qrels.values() for d in rels})
    others = sorted(set(docs) - set(relevant))
    rng = np.random.default_rng(SEED)
    extra = rng.choice(others, RETRIEVAL_CORPUS_SIZE - len(relevant), replace=False)
    corpus = {
        i: DOC_TEMPLATE.format(title=docs[i]["title"] or "none", text=docs[i]["text"])
        for i in [*relevant, *sorted(extra)]
    }
    return queries, corpus, qrels


def calibration_texts(n: int) -> list[str]:
    """Prefixed texts disjoint from the eval sets: STS-B train sentences,
    SciFact train queries and SciFact docs outside the eval corpus, in equal parts."""
    rng = np.random.default_rng(SEED)
    sts = load_stsb("train").sentence1.drop_duplicates().tolist()
    train_ids = {str(r["query-id"]) for r in _jsonl(_fetch("mteb/scifact", "qrels/train.jsonl"))}
    queries = [q["text"] for q in _jsonl(_fetch("mteb/scifact", "queries.jsonl")) if str(q["_id"]) in train_ids]
    _, eval_corpus, _ = load_scifact()
    docs = [
        DOC_TEMPLATE.format(title=d["title"] or "none", text=d["text"])
        for d in _jsonl(_fetch("mteb/scifact", "corpus.jsonl"))
        if str(d["_id"]) not in eval_corpus
    ]
    k = n // 3
    pick = lambda xs, m: [xs[i] for i in rng.choice(len(xs), m, replace=False)]
    return (
        [STS_PREFIX + t for t in pick(sts, k)]
        + [QUERY_PREFIX + t for t in pick(queries, k)]
        + pick(docs, n - 2 * k)
    )


def ndcg_at_10(scores: np.ndarray, qids: list, dids: list, qrels: dict) -> float:
    """Mean NDCG@10 with linear gain (as pytrec_eval / BEIR)."""
    discount = 1 / np.log2(np.arange(2, 12))
    vals = []
    for row, qid in enumerate(qids):
        top = np.argsort(-scores[row])[:10]
        gains = np.array([qrels[qid].get(dids[j], 0) for j in top])
        ideal = np.sort(list(qrels[qid].values()))[::-1][:10]
        vals.append((gains * discount[: len(gains)]).sum() / (ideal * discount[: len(ideal)]).sum())
    return float(np.mean(vals))


def evaluate(enc: Encoder) -> dict[str, float]:
    """Encode once at 768-d and derive every MRL dim by truncation."""
    sts = load_stsb()
    a, b = enc(list(sts.sentence1), STS_PREFIX), enc(list(sts.sentence2), STS_PREFIX)
    queries, corpus, qrels = load_scifact()
    q, d = enc(list(queries.values()), QUERY_PREFIX), enc(list(corpus.values()))

    out = {}
    for dim in MRL_DIMS:
        cos = (truncate(a, dim) * truncate(b, dim)).sum(1)
        out[f"sts_spearman_{dim}"] = float(spearmanr(cos, sts.score).statistic)
        sims = truncate(q, dim) @ truncate(d, dim).T
        out[f"scifact_ndcg10_{dim}"] = ndcg_at_10(sims, list(queries), list(corpus), qrels)
    return out
