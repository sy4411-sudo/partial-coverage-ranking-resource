# Evaluate a received policy

This downstream recipe completes a research task: receive a policy, verify it
against your own request, attach evaluation-only gains, calculate expected nDCG,
and sample the distribution. It uses only the standard library and the resource's
checker. The optimizer never receives the gain file.

## Connect the text retrieval workflow

From a checkout or source archive containing this example, with the package
installed, run these commands in separate processes:

```sh
python -I examples/retrieval_workflow.py --output workflow-output
python -I examples/downstream_evaluation.py --requests workflow-output/requests.json --policies workflow-output/policies.json --gains examples/retrieval/illustrative_gains.json --output workflow-output/evaluation.json --draws 10000 --seed 0
```

The supplied gain file contains **invented, illustrative assignments**, chosen
to exercise graded evaluation on the default four-query candidate sets. They
are not human relevance judgments, held-out labels, or evidence of retrieval
quality. They were added after developing the producer, not preregistered.
The producer still has no gain-file argument. Its `labels_used: false` receipt
describes only production; the separate evaluation report does use gains.

The report includes exact mixture-weighted expected nDCG, its difference from
the reference, an exposure-based cross-check, sampled component counts and
mean nDCG, and checks of two deterministic replacements. "Exact" here means
enumerating the finite mixture instead of Monte Carlo estimation; arithmetic
is floating point. No natural-domain effect is inferred from these inputs.

## Five hand-computable cases

```sh
python -I -m metric_aligned_ranking.run_adapter optimize examples/evaluation/requests.json > boundary-policies.json
python -I examples/downstream_evaluation.py --requests examples/evaluation/requests.json --policies boundary-policies.json --gains examples/evaluation/gains.json --output boundary-evaluation.json
```

Use a shell that writes redirected JSON as UTF-8 (PowerShell 7 or a POSIX shell).
On Windows PowerShell 5, replace the first command's redirection with
`| Out-File -Encoding utf8 boundary-policies.json`.

Every reference is `[a, fixed, b]` and the cutoff is one. In the first three
cases source scores favor `b`; `fixed` cannot move. With budget `p`, the emitted
policy places `b` first with probability `p`. Values below follow directly from
these constructed inputs and are executable checks, not measured user benefits.

| Case | Budget | Positive gain | Reference nDCG | Expected nDCG | Deterministic replacements |
| --- | ---: | --- | ---: | ---: | --- |
| reference_gain | 0.6 | a | 1 | 0.4 | Both fail the original budget |
| source_gain | 0.6 | b | 0 | 0.6 | Both fail the original budget |
| small_budget | 0.2 | b | 0 | 0.2 | Both pass, but nDCG becomes 0 |
| no_coverage | 0.6 | a | 1 | 1 | Reference fallback |
| zero_gain | 0.6 | none | 0 | 0 | Metric convention does not remove certificate checks |

The replacements are the largest-probability component (first component on a
tie) and a descending expected-exposure sort **within supported slots** (reference
order on a tie). Each is checked as a new deterministic policy. Keeping slots
fixed isolates the regret issue from a simpler coverage violation. A replacement
can pass and still change the endpoint: re-verifying it does not make it the
original randomized policy. A single draw can have regret exceeding the mixture
budget; the guarantee is an expectation over the mixture, not over a finite
batch of draws.

## Use your own evaluation inputs

The gain JSON maps query IDs to candidate-ID/nonnegative-gain objects. It must
contain exactly the queries and all candidates in the separately retained
requests. Missing judgments are rejected, not silently assigned zero. Explicit
zero is a judgment. If your dataset has incomplete qrels, decide and justify a
different evaluation protocol before adapting it; this recipe does not estimate
missing relevance or evaluate the unretrieved corpus.

Inputs are already gains, not relevance grades: convert grades explicitly if
your metric uses `2**grade - 1`. nDCG uses the supplied position weights and
an ideal ordering of this candidate universe. With the adapter's logarithmic
weights this is nDCG at its cutoff; custom weights define the corresponding
weighted normalized DCG. Gains are rescaled per query before arithmetic to
avoid overflow. All-zero gain queries receive zero and remain in the macro
average, with their count reported. Extreme gain ratios remain subject to
floating-point resolution.

The sampling seed creates a reproducible per-query stream in the tested Python
runtime. It does not guarantee identical draws across all Python versions.
Counts are diagnostics, not confidence intervals or a live-sampler audit.
The script refuses an existing output file and records hashes of all three
inputs. Keep requests in recipient-controlled storage; hashes do not authenticate
the sender. The consumer neither imports SciPy nor checks optimality.

## What the wrapper saves you from implementing

The [direct-solver comparison](../direct_solver_comparison.md) documents the
same-core alternative: map IDs, recover and serialize full permutations, retain
the original context, and check the received object. The direct solver already
returns mixtures; this is not a new optimizer. Tests feed both paths through
this consumer on the five cases above and recover the same known endpoints.
This is a comparison of integration responsibilities, not a measurement of
saved hours, a second independent algorithm, or evidence of researcher adoption.

The example is appropriate for fully judged candidate sets and experiments
that execute stochastic rankings. It does not turn a deterministic ranking API
into a stochastic one, provide click-propensity correction, certify a deployment
sampler, or recover labels from a closed historical experiment.
