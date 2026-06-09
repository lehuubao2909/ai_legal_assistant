"""Reranker-score cutoff — pure python, NO heavy deps.

Shared by local_rag_engine (live retrieval), generate_submission (--retrieved cache),
and scratch/sweep_cutoff.py (offline leaderboard-variant sweeping). Keeping it dependency-
free lets the submission/sweep paths import it without pulling sentence-transformers.

Cutoff policy (recall-floor): always keep the top-1 candidate, then keep each next one
while it passes an absolute floor (min_score) AND stays within `margin` of the top score,
capped at top_k. F2 favors recall 2× — prefer a gentle (loose) cutoff over an aggressive one.
"""
from typing import List, Dict, Any, Optional


def apply_cutoff(
    ranked: List[Dict[str, Any]],
    top_k: int,
    min_score: Optional[float] = None,
    margin: Optional[float] = None,
    score_key: str = "score",
) -> List[Dict[str, Any]]:
    """Trim a score-sorted candidate list. `ranked` must be sorted desc by score_key.

    - top_k     : hard cap on returned count.
    - min_score : drop candidates with score < this (None = no absolute floor).
    - margin    : drop candidates with score < (top_score - margin) (None = no relative cut).
    """
    if not ranked:
        return []
    out = [ranked[0]]
    top = ranked[0].get(score_key, 0.0)
    for d in ranked[1:top_k]:
        s = d.get(score_key, 0.0)
        if min_score is not None and s < min_score:
            break
        if margin is not None and s < top - margin:
            break
        out.append(d)
    return out
