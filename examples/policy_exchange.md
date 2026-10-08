# Export and verify a ranking policy

The `policy_io` interface accepts your string candidate IDs and returns a JSON
ranking mixture. The verifier reconstructs exposure and checks expected-nDCG
regret using only Python's standard library and this package. It does not
need SciPy, source scores, labels, a dataset, or historical experiment receipts.

Run these commands from the repository root. Python 3.11 or newer is required.
If using the standalone trial ZIP instead of a checkout, follow
`START_HERE_ZH.md` for wheel installation; the ZIP is not an editable source
installation. The concepts and Python API below apply to either installation.

```powershell
python -m pip install -e ".[solver]"
python -m metric_aligned_ranking.policy_io optimize examples/policy_context.json examples/policy_scores.json
python -S -m metric_aligned_ranking.policy_io verify examples/policy_mixture.json examples/policy_context.json
```

The first command after installation prints a newly optimized policy to
stdout. The last checks the separately supplied, hand-written fixture, whose
regret is approximately `0.1`; it does not claim those mixture bytes must be
the solver's unique output. `-S` disables site packages, demonstrating that
verification can run from the source checkout without the optimizer stack.
For an installed package outside the checkout, omit `-S`.

## Python round trip

```python
import json
from pathlib import Path
from metric_aligned_ranking.policy_io import (
    loads_policy_json,
    optimize_policy,
    verify_policy,
)

context = loads_policy_json(Path("examples/policy_context.json").read_text())
scores = loads_policy_json(Path("examples/policy_scores.json").read_text())
policy = optimize_policy(context, source_scores=scores)
encoded = json.dumps(policy, allow_nan=False)
received = loads_policy_json(encoded)
report = verify_policy(received, expected_context=context)
assert report["valid"]
print(report["regret"])
```

`context` must come from the caller's intended task, not be copied out of an
untrusted policy to make it pass. Keep it separately when sending a policy
to another process. The checker compares target order, supported IDs,
position weights and budget before checking the mixture. This does not
authenticate a sender or protect a context that was itself replaced.

## Input and output meaning

The context has exactly four fields:

| Field | Meaning |
|---|---|
| `target_ranking` | Complete ranking of unique, nonempty string IDs |
| `supported_items` | Unique subset with source scores; empty support is allowed |
| `position_weights` | Positive, non-increasing weights through the cutoff; remaining slots have zero weight |
| `risk_budget` | Nonnegative absolute worst-case expected-nDCG loss relative to the target |

The separate score mapping has exactly the supported IDs as keys and finite
numeric values. Zero scores and negative scores are allowed. Unsupported
items retain their original positions in every mixture component. The policy
contains its schema, context, and a list of full rankings with probabilities.
IDs may reveal private information: the example IDs are invented, but your
own policies should not be published without an appropriate privacy review.

`optimize_policy(..., formulation="cutting-plane")` selects the existing
alternative formulation; `"compact"` is the default. The CLI accepts the same
choice through `--formulation`. Objective-equivalent solutions need not have
identical mixture bytes. In particular, tied source scores do not promise a
deterministic target-policy tie break. Solver failures raise
`PolicyOptimizationError`; malformed public inputs raise `PolicyFormatError`.
Neither a timeout nor a numerical failure establishes infeasibility.

Since package 0.1.1, verification reports also list `supported_items` and
`unsupported_items` from the checked context. These are coverage facts, not
predictions of relevance or an instruction to impute missing scores. The
policy JSON schema and numerical acceptance rules are unchanged. Diagnostics
include numeric mass or regret values; batch errors additionally identify
the query. Error messages and reports can contain your IDs, so share only
synthetic or suitably redacted reproductions.

The verifier rejects unknown fields, duplicate JSON keys, malformed IDs,
non-finite numbers, changed references, unsupported-slot movement, malformed
probability mass, and regret above the supplied budget plus `1e-9`. CLI success
returns exit code 0; rejected inputs return 2 and an error on stderr.

## Sampling and numerical scope

Probabilities must sum to one within `1e-12`; the represented policy uses each
probability divided by that sum. The report records the original sum. Larger
mass errors are rejected rather than repaired. The reconstructed regret uses
floating-point arithmetic and an absolute acceptance tolerance of `1e-9`.
There is no claim of interval-arithmetic verification.

To draw from an already verified policy, sample the full ranking mixture:

```python
import random

rng = random.Random(7)  # Example reproducibility seed, not a production rule.
ranking = rng.choices(
    [entry["ranking"] for entry in received["mixture"]],
    weights=[entry["probability"] for entry in received["mixture"]],
    k=1,
)[0]
```

`random.choices` uses relative weights, agreeing with the normalized policy.
Do not replace the mixture with its most probable ranking or alter a ranking
after verification and retain the same claim. The certificate covers expected
policy regret for arbitrary nonnegative gains, not each realized draw. It
does not show source-score quality, empirical improvement, fairness,
optimality, or that a live system really sampled the verified distribution.

The mathematical checker is shared with the existing implementation; it is
solver-independent, not a separate proof system. Tiny exhaustive tests check
it against all nonempty binary relevance sets. No frozen evaluation is run
or reopened by this example.

For an executable comparison with direct lower-level solver assembly, see
[what the policy interface adds](direct_solver_comparison.md). That comparison
does not claim a different algorithm or measured developer-time savings.
