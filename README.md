# Metric-aligned ranking: a portable policy resource

**Version 0.2.0.** By Shenghang Yuan, NYU Shanghai.
[ORCID](https://orcid.org/0009-0008-7434-1103).
[Versioned release](https://github.com/sy4411-sudo/partial-coverage-ranking-resource/releases/tag/v0.2.0).

A Python research tool for reranking a partially scored candidate set, exporting
a distribution over rankings, and checking that distribution against a separately
supplied reference and worst-case **expected** nDCG-regret budget.

This is a clean, allowlisted software distribution, not a copy of the private
research repository. No private feedback, raw datasets, user IDs, individual
effects, credentials, or experimental execution authorizations are distributed.

## Start with an actual retrieval workflow

Use Python 3.12 in a fresh virtual environment. From an extracted release or checkout:

```sh
python -m pip install ".[retrieval,test]"
python -I examples/retrieval_workflow.py --output workflow-output
python -I -m metric_aligned_ranking.run_adapter verify workflow-output/policies.json workflow-output/requests.json
python -m pytest -q
```

[The walkthrough](examples/retrieval/README.md) takes text through BM25 retrieval,
budget-limited TF-IDF scoring, TREC export, optimization, serialization and
recipient verification. It includes invented text and supports your own inputs.
It is a runnable research integration, not a new relevance benchmark or human reuse study.

## Smaller examples and interfaces

- [Policy exchange](examples/policy_exchange.md): JSON format, verifier and trust boundary.
- [Run-file adapter](examples/run_files.md): missing scores remain unsupported.
- `python -I examples/trial_workflow.py --output toy-output`: two-query TREC example.
- `python -I examples/direct_solver_comparison.py --self-check`: nine wrapper/direct-solver equivalence cases.
- `python -I scripts/analysis/summarize_resource_benchmark.py`: reproduce engineering tables without solving.
- `python -m metric_aligned_ranking.resource_benchmark --help`: rerun synthetic CPU workloads.

Only solving requires SciPy. Install the wheel with `--no-deps` to use the
pure-Python verifier. Run `python -I -m metric_aligned_ranking.policy_io verify
examples/policy_mixture.json examples/policy_context.json`.

## What the guarantee does and does not say

Unsupported items keep their reference slots in every mixture component.
The certificate bounds expected nDCG regret for arbitrary nonnegative gains on
the supplied candidate universe. It applies to the normalized mixture, **not**
every sampled list, an argmax component, or a sort of expected exposure.
Verification does not establish relevance, optimality, deployment correctness,
authenticity of the reference, or statistical calibration. Floating-point
acceptance uses regret tolerance 1e-9 and probability-mass tolerance 1e-12.

## Evidence and paper

[Complete resource manuscript](paper/resource/main.tex) and the release PDF
describe the resource, interfaces, guarantee scope, limitations and bounded case.
This is an **unsubmitted author manuscript**, not an accepted ECIR paper.

The saved engineering sweep has 26 workload/formulation combinations and 78
repetitions on one Windows machine. Measurements were made with implementation
0.1.0, not relabeled as 0.2.0 timings. The release adds a retrieval integration;
the underlying optimizer and certificate oracle are unchanged.
[Evidence notes](EVIDENCE.md) explain which records can be regenerated and
which are read-only historical aggregates. Human independent reuse remains pending.

## Release integrity and maintenance

The release includes a source ZIP, installable wheel, manuscript PDF and
SHA256SUMS. The source manifest hashes every exported payload file; it is a
content inventory, not a cryptographic signature or immutable preservation
service. GitHub releases are versioned but maintainers can replace them.
No DOI or PyPI publication is claimed. Preserve a downloaded release with its hashes.

Code, invented examples and documentation: Apache-2.0, see LICENSE and NOTICE.
Upstream packages retain their own licenses; no third-party source is vendored.
Use CITATION.cff for software citation. Open a minimal synthetic reproduction in
Issues; never upload private datasets or identifiers. There is no production SLA.
AI assistance was used in development and manuscript drafting; author review and
responsibility remain necessary.
