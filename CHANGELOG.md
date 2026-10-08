# Changelog

## 0.3.0 - 2026-10-08

Adds a solver-free downstream evaluation recipe, separate illustrative gains
for the text workflow, five hand-computable policy-use cases, and tests of both
public-interface and direct-core outputs at the evaluation boundary. The new
consumer checks complete judgments, calculates expected metrics and reports
sampling diagnostics. It rechecks deterministic replacements as new policies.
Adds the previously omitted direct-solver comparison guide to the source
allowlist. No optimizer, verifier, historical study or released asset changes.
The accompanying revised manuscript remains unsubmitted; it does not replace
the 0.2.1 release manuscript or assert new natural-domain effectiveness.

## 0.2.1 - 2026-10-08

The portable policy interface now requires the first position weight to equal
1.0. Version 0.2.0 accepted arbitrary positive overall scales, allowing exposure
underflow to falsely accept a zero budget for a mixture with actual regret 0.4
when its sole weight was 5e-324. Both optimization requests and recipient checks
now reject nonunit scales before exposure arithmetic. Callers must normalize
custom weights before constructing their trusted context. The JSON field layout
is unchanged, but this is an intentionally stricter input-acceptance rule.

Adds maintenance/compatibility guidance and a discussion of coverage and
reference bias. No historical optimizer, oracle, experiment or release asset is
changed. These corrections are released separately from 0.2.0; its assets remain
unchanged. Upgrade to 0.2.1 and re-verify any custom policies using a caller-owned
context whose first position weight is 1.0.

## 0.2.0 - 2026-10-07

First clean public resource release. Adds a text-to-policy integration using
BM25 and budget-limited TF-IDF cosine scores, replaceable corpus/query inputs,
provenance, tests, complete LNCS manuscript and allowlisted evidence distribution.

The optimizer, regret oracle and existing policy schema are carried over from
the private 0.1.1 preparation. No new scientific efficacy claim or changed
held-out evaluation is included. Historical timing records describe 0.1.0.

## Earlier preparation

0.1.1 improved failure diagnostics and coverage reporting. 0.1.0 introduced
the portable policy and run-file interfaces and synthetic engineering sweep.
These preparation versions were not public releases of this repository.
