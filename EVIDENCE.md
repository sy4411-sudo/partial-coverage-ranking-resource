# Evidence boundaries

## Reproducible software and engineering tables

The source includes the current optimizer, recipient checker, TREC adapter,
retrieval example, portable regression tests and synthetic benchmark harness.
Tests include exhaustive small relevance assignments, invalid-policy rejection,
fixed unsupported slots, zero-budget cases, run-file identity checks and exported
policy round trips. Automated checks are not independent human reuse.

`records/resource/engineering-resource-v2-20261007.json` is the original synthetic
engineering measurement record from software 0.1.0. It has no raw user data.
Its source digests identify the measured implementation. Release 0.2.0 does not
claim its newer diagnostics and retrieval example were timed in that record.
The summary script regenerates descriptive tables from the record; rerunning
the harness produces new timings, not bit-identical historical measurements.
The matched-support cases were exploratory corrections to an initial confounded
scaling comparison, not a preregistered experiment.

## Bounded historical case

`records/case-study-summary.json` is an explicitly selected aggregate export
from a previously terminal train-only MIND-small study. It contains decision
rows and terminal diagnostics, not labels, impressions or participant IDs.
Its source SHA-256 identifies the private original aggregate bytes, not a
publicly available full provenance chain. This software release does not
independently reproduce that closed study. No held-out stream was reopened.

The case concerns six target-blocked pairs and three risk levels evaluated on
the same 512 final impressions. It cannot establish a domain-wide limitation,
a per-impression risk-violation rate, or the distribution of the earlier pilot.
No observed positive mean-risk excess was avoided in these comparisons;
measurable utility differences are not thereby measured certificate value.

## What remains missing

A real person independently adapting this resource to their own research
workflow, independently measured time savings, and multi-machine performance
evidence are not supplied. They are not inferred from agent-run checks.
The manuscript is a complete draft, not a claim that every submission weakness
has been resolved. No prior negative scientific conclusion has been reversed.
