# Rebuttal playbook

## Goal

Produce a concise, evidence-backed, point-by-point response while keeping code,
configuration, experiments, manuscript revisions, and response claims
synchronized.

## Required inputs

- Verbatim reviewer comments in `rebuttal/reviewer_comments.md`.
- Venue response rules, including global or per-reviewer word/character limit.
- The current paper, code, configs, and available run artifacts.
- Any deadline, compute budget, or experiment restrictions supplied by the
  authors.

If a required input is missing, useful triage may proceed, but no agent should
invent the missing material.

## Issue identifiers and states

Assign stable IDs such as `R1.C1`, `R1.C2`, and `R2.C1`. Track one row per
substantive concern in `rebuttal/response_matrix.md`.

Allowed states:

`NEW -> TRIAGED -> EVIDENCE_NEEDED -> IMPLEMENTING -> DRAFTED -> VERIFIED -> DONE`

Use `BLOCKED` when a required dataset, artifact, author decision, or compute
resource is unavailable.

## Orchestration

1. **Intake:** Preserve comments verbatim and record response limits.
2. **Strategy:** `rebuttal_strategist` decomposes compound comments, classifies
   concern type and risk, maps existing evidence, and proposes the smallest
   decisive action. The primary agent records accepted decisions in the matrix.
3. **Execution:** Delegate independent implementation and manuscript tasks.
   `rebuttal_code` owns code/config/tests; `rebuttal_paper` owns verified
   manuscript changes. Do not let both edit the same file.
4. **Verification:** The primary agent checks code behavior, experiment
   evidence, paper wording, tables, citations, and LaTeX. A claim is verified
   only when its evidence location and validation are recorded.
5. **Writing:** `rebuttal_writer` drafts the response only from verified matrix
   rows. The primary agent checks completeness, tone, consistency, and budget.

Strategy and evidence collection can run in parallel across independent
reviewer comments. Final response writing is sequential after evidence and
paper/code changes stabilize.

## Concern taxonomy

- **Correctness:** equations, implementation, assumptions, metric definitions.
- **Novelty/positioning:** distinction from robust losses, B-PINN, anomaly
  detection, or selective learning.
- **Experimental evidence:** baselines, seeds, uncertainty, ablations,
  sensitivity, compute, or missing comparisons.
- **Reproducibility:** configs, data generation, hyperparameters, seeds,
  artifact traceability.
- **Clarity:** terminology, motivation, algorithm, captions, appendix pointers.
- **Scope/limitation:** generality, failure modes, boundary conditions,
  applicability, or overclaiming.

## Evidence hierarchy

Prefer, in order:

1. Existing verified paper table/figure with traceable artifact.
2. Existing run outputs and configs that directly answer the concern.
3. A targeted new experiment with recorded command, seed, artifact, and
   aggregation method.
4. A code-level explanation supported by tests and precise source locations.
5. A manuscript clarification or explicit limitation.

Do not promise a new experiment unless it can finish and be verified within the
available compute and rebuttal deadline.

## Response pattern

For each issue:

1. Thank or acknowledge the substantive concern without filler.
2. State the direct answer in the first sentence.
3. Give the minimum decisive evidence, including numbers only when verified.
4. State exactly what changed in the manuscript/code, with section or artifact.
5. State a limitation honestly when the concern cannot be fully resolved.

Avoid reviewer restatement, defensive language, unsupported superlatives,
future-work promises presented as evidence, and claims that a smoke test
reproduces a paper result.

## Final gates

- Every substantive reviewer point has one matrix row and one response.
- Every number in the response matches a verified source.
- Every claimed revision exists in the current paper or code diff.
- Main paper terminology matches implementation conventions.
- New experiments include command, config, seeds, hardware, outputs, and
  aggregation details.
- The paper compiles or the missing TeX toolchain is explicitly reported.
- The response meets the exact venue limit.
- `docs/PROGRESS.md` reflects the final verified state.

## Suggested Codex request

```text
Run the naPINN rebuttal workflow. Read AGENTS.md and the rebuttal documents,
delegate strategy, code, and paper work only where independent, verify every
claim, and draft the final response within the recorded venue limit.
```
