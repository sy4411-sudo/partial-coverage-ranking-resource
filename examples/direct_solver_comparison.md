# What the policy interface adds to the solver

The existing lower-level solver already returns ranking mixtures, not just
an exposure vector. The resource interface does not invent policy recovery
or improve that solver's objective. It standardizes the input, exchange and
checking work that callers otherwise assemble around it.

## From the trial ZIP

If you have the separate legacy trial packet, complete its `START_HERE_ZH.md`
installation and run these commands from that packet. The public source archive
does not contain that trial handout; use the source-checkout instructions below.
No repository checkout or pytest is needed for the legacy packet:

```powershell
.\.venv\Scripts\python.exe examples/direct_solver_comparison.py
.\.venv\Scripts\python.exe examples/direct_solver_comparison.py --self-check
```

On Linux use `./.venv/bin/python`. The first command runs the bundled fixture;
`--self-check` runs all nine built-in settings and prints `case_count: 9`.

## From a source checkout

Only if using the full repository, install its solver extra and run:

```powershell
python -m pip install -e ".[solver]"
python examples/direct_solver_comparison.py
```

The [example](direct_solver_comparison.py) takes the same invented context
through two paths. The public path calls `optimize_policy`. The explicit path
maps string IDs to integer indices, calls `optimize_robust_slot_policy_compact`,
maps the returned rankings back to the original IDs and assembles the public
JSON object. Both serialize and parse the object, then verify it against the
original context. Neither path derives the trusted context from the received
policy. Both use the same core optimizer and mathematical checker.

The example requires feasible policies and source-objective agreement within
`1e-8`, not byte-identical mixtures. Nine small test settings cover no support,
one supported item and partial support at budgets zero, 0.1 and one. These
are interface regression tests, not an independent algorithm comparison.

| Responsibility | Direct solver caller | Public resource interface |
|---|---|---|
| Candidate identity | Build and retain integer-to-string mapping | Accepts external string IDs |
| Coverage | Construct support and matching score keys | Validates the explicit subset and score keys |
| Policy recovery | Already provided by the core solver | Reuses the same recovered mixture |
| Exchange | Assemble a representation and keep reference context separately | Documented JSON representation and strict parser |
| Recipient checking | Reconstruct exposure and supply the intended reference | Standard-library verifier requiring a separate context |
| Multiple run-file queries | Parse, group, map, solve and check each query | TREC adapter preserving missing rows and exact query set |

The explicit path intentionally assumes the fixture was already validated;
it is not a second supported general-purpose input validator. A caller can
compose the existing components to reproduce the resource behavior. This
comparison therefore demonstrates concrete integration responsibilities,
not irreducibility, theoretical novelty, fewer programmer hours or an
advantage over third-party tools. Actual reuse feedback is needed to learn
which responsibilities matter to another researcher.

## Reading and checking results

`valid` means both exported policies passed the same feasibility checker and
their source objectives agreed to the stated tolerance. It does not certify
relevance gains or optimality independently. Source-objective agreement is
expected because the two paths invoke the same solver.

The [downstream recipe](evaluation/README.md) extends the comparison beyond
verification: both paths feed the same solver-free evaluator on five constructed
cases with known expected nDCG. It also connects to the text-retrieval output.
It does not assume identical mixtures for arbitrary source-objective ties.

## Engineering table from the repository only

The trial ZIP deliberately omits the benchmark reader and measured records.
The following commands require a full repository checkout; they are not
additional trial steps. They regenerate the existing engineering table
without running any optimizer:

```powershell
python scripts/analysis/summarize_resource_benchmark.py
python scripts/analysis/summarize_resource_benchmark.py --format json
```

The reader includes all 26 recorded workload/formulation combinations,
reports missing memory measurements as `n/a`, and preserves failed workload
statuses rather than treating them as zero-time successes. Timings are
within-workload medians; formulation objective comparisons use the median
source utility on matching complete workload definitions. This command reads
the saved synthetic record only and does not repeat the benchmark or any
closed evaluation.
