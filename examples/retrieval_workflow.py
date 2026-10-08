"""BM25 -> budgeted TF-IDF scoring -> exported, independently checked policies.

The bundled text is invented. This is an integration example, not a relevance
evaluation. No qrels, training labels, network access, or model downloads occur.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
from pathlib import Path

from metric_aligned_ranking.policy_io import loads_policy_json
from metric_aligned_ranking.run_adapter import optimize_runs, prepare_runs, verify_runs


def read_texts(path):
    rows = loads_policy_json(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path.name}: expected a nonempty list")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "text"}:
            raise ValueError(f"{path.name}: rows require exactly id and text")
        item = row["id"]
        if not isinstance(item, str) or not item or re.search(r"\s", item):
            raise ValueError(
                f"{path.name}: IDs must be nonempty whitespace-free strings"
            )
        if item in seen:
            raise ValueError(f"{path.name}: duplicate ID {item}")
        seen.add(item)
        if not isinstance(row["text"], str) or not re.findall(r"\w+", row["text"]):
            raise ValueError(f"{path.name}: text must contain a token")
    return rows


def make_runs(corpus, queries, *, candidates=12, source_limit=6):
    from rank_bm25 import BM25Okapi
    from sklearn.feature_extraction.text import TfidfVectorizer

    if candidates < 1 or not 0 <= source_limit <= candidates:
        raise ValueError("require candidates >= 1 and 0 <= source-limit <= candidates")

    def tokenize(text):
        return re.findall(r"\w+", text.lower())

    bm25 = BM25Okapi([tokenize(row["text"]) for row in corpus])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b")
    documents = vectorizer.fit_transform([row["text"] for row in corpus])
    target, source, counts = [], [], {}
    for query in queries:
        scores = bm25.get_scores(tokenize(query["text"]))
        order = sorted(
            range(len(corpus)), key=lambda i: (-float(scores[i]), corpus[i]["id"])
        )[:candidates]
        supported = order[:source_limit]
        # Only supported candidates receive query-document second-stage scores.
        qvector = vectorizer.transform([query["text"]])
        partial = (
            (documents[supported] @ qvector.T).toarray().ravel() if supported else []
        )
        for rank, i in enumerate(order, 1):
            target.append(
                f"{query['id']} Q0 {corpus[i]['id']} {rank} {float(scores[i]):.17g} bm25"
            )
        for rank, (i, score) in enumerate(zip(supported, partial), 1):
            source.append(
                f"{query['id']} Q0 {corpus[i]['id']} {rank} {float(score):.17g} tfidf"
            )
        counts[query["id"]] = {
            "candidates": len(order),
            "second_stage_scores": len(supported),
        }
    return (
        "\n".join(target) + "\n",
        "\n".join(source) + ("\n" if source else ""),
        counts,
    )


def run(
    corpus_path,
    queries_path,
    output,
    *,
    candidates=12,
    source_limit=6,
    cutoff=5,
    budget=0.02,
):
    if output.exists():
        raise ValueError("output already exists; use a new directory")
    if cutoff < 1 or not math.isfinite(budget) or budget < 0:
        raise ValueError("require cutoff >= 1 and a finite nonnegative budget")
    corpus, queries = read_texts(corpus_path), read_texts(queries_path)
    target, source, counts = make_runs(
        corpus, queries, candidates=candidates, source_limit=source_limit
    )
    requests = prepare_runs(target, source, cutoff=cutoff, risk_budget=budget)
    policies = optimize_runs(requests)
    output.mkdir(parents=True)
    (output / "target.trec").write_text(target, encoding="utf-8")
    (output / "source_partial.trec").write_text(source, encoding="utf-8")
    for name, value in (("requests", requests), ("policies", policies)):
        (output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    # Receiver rebuilds context from upstream runs, not from untrusted policy.context.
    expected = prepare_runs(
        (output / "target.trec").read_text(),
        (output / "source_partial.trec").read_text(),
        cutoff=cutoff,
        risk_budget=budget,
    )
    received = loads_policy_json((output / "policies.json").read_text())
    report = verify_runs(received, expected_requests=expected)
    (output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema": "ranking.retrieval-workflow.v1",
        "scope": "software integration; no relevance evaluation",
        "input_sha256": {
            "corpus": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
            "queries": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        },
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("rank-bm25", "scikit-learn", "scipy", "numpy")
        },
        "parameters": {
            "candidates": candidates,
            "source_limit": source_limit,
            "cutoff": cutoff,
            "budget": budget,
        },
        "counts": counts,
        "valid": report["valid"],
        "labels_used": False,
        "outputs_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(output.iterdir())
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent / "retrieval"
    parser.add_argument("--corpus", type=Path, default=root / "corpus.json")
    parser.add_argument("--queries", type=Path, default=root / "queries.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--source-limit", type=int, default=6)
    parser.add_argument("--cutoff", type=int, default=5)
    parser.add_argument("--budget", type=float, default=0.02)
    args = parser.parse_args()
    try:
        result = run(
            args.corpus,
            args.queries,
            args.output,
            candidates=args.candidates,
            source_limit=args.source_limit,
            cutoff=args.cutoff,
            budget=args.budget,
        )
    except (ValueError, OSError) as exc:
        parser.exit(2, f"retrieval workflow error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
