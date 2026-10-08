# Use partial source scores from run files

This adapter accepts six-column TREC run files, the format supported by
[PyTerrier's result writer](https://pyterrier.readthedocs.io/en/stable/io.html).
It does not require PyTerrier, Java, pandas or qrels. The bundled two-query
example uses invented document and query IDs only.

If you have the trial ZIP, start with `START_HERE_ZH.md` and its installed-wheel
commands. The repository CLI and engineering sweep below require files that
are not all included in the trial ZIP.

Each nonempty line must contain:

```text
query-id Q0 document-id ordinal-rank numeric-score run-tag
```

Target order follows the explicit ordinal ranks, not line order or score
sorting. Ranks may start at zero or one and have gaps, but cannot repeat
within a query. The source file supplies scores only for covered documents;
its rank column is parsed but does not define the target reference.

## Python workflow

```python
from pathlib import Path
from metric_aligned_ranking.run_adapter import (
    prepare_runs, optimize_runs, verify_runs,
)

requests = prepare_runs(
    Path("examples/target.trec").read_text(encoding="utf-8"),
    Path("examples/source_partial.trec").read_text(encoding="utf-8"),
    cutoff=2,
    risk_budget=0.1,
)
batch = optimize_runs(requests)
report = verify_runs(batch, expected_requests=requests)
assert report["query_count"] == 2
assert report["reports"]["toy-2"]["regret"] == 0
```

`toy-1` has four target candidates but only two source scores. `toy-2` has no
source scores and returns its target policy. Source rows with score zero are
covered; missing source rows are not silently filled with zero. Source-only
queries or candidates are rejected rather than extending the candidate set.
Cutoff is truncated to the available candidate count for short lists.

The verifier checks the full query set as well as each individual policy.
It rejects missing or extra queries. Keep the original requests separately:
rebuilding expected requests from an untrusted policy defeats reference
checking. The batch verifier validates the request structure but does not
certify the quality of source scores or optimize an empirical metric.

## Command line workflow

From the repository root, after installing `.[solver]`, these PowerShell
commands write into a new scratch directory. Keep your actual run files and
their query/document identifiers out of public commits.

```powershell
$demo = Join-Path ([IO.Path]::GetTempPath()) ("ranking-demo-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $demo | Out-Null
python -m metric_aligned_ranking.run_adapter prepare examples/target.trec examples/source_partial.trec --cutoff 2 --risk-budget 0.1 | Set-Content -Encoding utf8 (Join-Path $demo "requests.json")
python -m metric_aligned_ranking.run_adapter optimize (Join-Path $demo "requests.json") | Set-Content -Encoding utf8 (Join-Path $demo "policies.json")
python -S -m metric_aligned_ranking.run_adapter verify (Join-Path $demo "policies.json") (Join-Path $demo "requests.json")
```

Check the exit status before using output from any stage. The library API is
preferable for applications that need explicit exception handling. Batch
optimization currently holds its requests and policies in memory and does
not supply per-query process timeouts. Use the benchmark's isolated worker
pattern for bounded experiments; do not treat this adapter as a production
queue or untrusted-input network service.

The output is a stochastic policy bundle, not a TREC file pretending that an
argmax ranking has the same certificate. Follow the
[sampling instructions](policy_exchange.md#sampling-and-numerical-scope) to
draw rankings. Standard deterministic run evaluators do not automatically
evaluate expected utility for a mixture.

## Reproduce the engineering sweep

```powershell
python -m metric_aligned_ranking.resource_benchmark --profile all --repeats 3 --timeout 20 --output outputs/resource-engineering-new.json
```

The output path must not already exist. The manifest runs 26 workload/solver
combinations with at most 20 seconds per child. It records failures and
timeouts as rows; a completed driver is not proof that every child passed.
Inspect row statuses. The sweep uses synthetic scores only and does not
reopen any historical evaluation.
