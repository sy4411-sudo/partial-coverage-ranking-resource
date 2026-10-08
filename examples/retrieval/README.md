# A text-to-policy research workflow

This example runs real BM25 (`rank-bm25`) and unigram/bigram TF-IDF cosine
(`scikit-learn`), then the resource's existing optimizer and recipient checker.
The 20 documents and four queries are invented and Apache-2.0 licensed.
There are no relevance labels and no retrieval-effectiveness claim.

From the public release root, in a fresh Python 3.12 environment:

```sh
python -m pip install ".[retrieval]"
python -I examples/retrieval_workflow.py --output workflow-output
python -I -m metric_aligned_ranking.run_adapter verify workflow-output/policies.json workflow-output/requests.json
```

BM25 retrieves 12 candidates per query. Only the top six receive second-stage
query-document cosine scores; their target slots may be exchanged. TF-IDF
features are fitted on the corpus, but second-stage similarities are computed
only for the covered subset. This cheap scorer demonstrates the boundary;
it is not a neural model or evidence of compute savings. A real pipeline can
replace that scorer while preserving the same TREC and policy interfaces.

The output includes both run files, caller-owned requests, serialized ranking
mixtures, independently recomputed reports, dependency versions and input/output
hashes. The receiver reconstructs the expected context from the upstream runs,
not from the received policy. In a real exchange, retain those inputs and the
budget in your own trusted storage; hashes are not sender authentication.
No result directory is overwritten. A failed run may leave partial files;
only a successful verification report establishes completion.

To use your own data, provide `--corpus corpus.json --queries queries.json`.
Each file is a nonempty JSON list of objects with exactly `id` and `text`.
IDs must be unique within a file and contain no whitespace. Supply only text
you are allowed to process and do not publish private inputs or generated runs.
Tokenization is deliberately simple and is not a multilingual retrieval recipe.
`--source-limit 0` demonstrates reference fallback; `--source-limit 12` covers
the whole candidate set. `--budget` is absolute worst-case expected-nDCG regret.

The primary output is a distribution over full rankings. Do not replace it with
the largest-probability ranking or a sort of expected exposure and retain the
same guarantee. The guarantee is in expectation over the emitted mixture,
for arbitrary nonnegative gains on the supplied candidate universe; it is not
a bound on every random draw, empirical accuracy, or the wider corpus.

For the next step, use the [downstream evaluation recipe](../evaluation/README.md).
It takes this workflow's saved requests and policies, attaches a separate
illustrative gain file, computes expected nDCG and samples the mixture without
importing a solver. The supplied assignments are invented, not relevance evidence.
