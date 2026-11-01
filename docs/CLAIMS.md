# Claims ledger

Principle 12 of the Plan: *do not build on false premises.* The research papers (P1-P7) contain many numeric and
qualitative claims ("RHT does not help NVFP4 RTN", "W4A16 keeps ~4x speed-up to batch 16-32", ...). The planner's
rules and the UI's recommendations depend on some of them. The **claims ledger** records each claim, how much we
trust it today, and which experiments would settle it, so that:

* a recommendation rests visibly on `untested` or `reproduced` evidence,
* experiment time goes first to claims that many rules depend on (Plan C-023),
* the UI never shows a paper number as if it were a measurement (provenance badges, Plan UI-3).

## Data model (`forge.core.schemas.Claim`)

| Field | Meaning |
|-------|---------|
| `id` | Stable id, e.g. `P1-F4` |
| `source_paper` | `P1`..`P7` (Quantization, Pruning, KV/attention compression, Distillation/low-rank/lossless, NVFP4, Sparse attention, Speculative decoding) |
| `location` | Section / table / finding in the paper |
| `statement` | The claim, phrased so it can be tested |
| `status` | `untested` -> `reproduced` / `contradicted` / `partial` |
| `prior` | `trusted` or `untrusted`: how much weight the planner gives it before testing |
| `linked_experiments` | Ids of eval reports / bench rows / jobs that tested it |
| `depends_rules` | Planner rules (rulebase entries) that rely on the claim |
| `notes` | Free text: conditions under which it was tested, caveats |

## Provenance tags

Every number shown in the UI carries a `Provenance`: `measured` (this system, this hardware), `reproduced`
(a paper's claim confirmed here), `paper_reported`, `derived` (computed from other numbers by a stated formula),
`vendor` (datasheet), `target` (a goal such as a KPI), `untested`. Paper numbers stay `paper_reported` until a
linked experiment upgrades them.

## Lifecycle

1. **Seed**: `forge.planner.claims.seed_claims()` loads the initial set (at least 30 claims from P1-P7), all
   `untested`, with `prior` set from the Plan's confidence ratings. Seeding is idempotent.
2. **Prioritise**: sort by number of `depends_rules` (Plan C-023). The ledger page shows "claims needing attention".
3. **Test**: run the experiment (eval suite, bench job, recipe). Link the resulting ids in `linked_experiments`.
4. **Resolve** (`PUT /api/v1/claims`): set `status`:
   * `reproduced` - the result agrees within the noise floor, under stated conditions;
   * `contradicted` - the result disagrees outside the noise floor;
   * `partial` - holds only in some regimes (record which in `notes`).

   Never change `status` without at least one linked experiment and a note.
5. **Propagate**: rules in `depends_rules` of a `contradicted` claim should be reviewed before the planner keeps
   recommending them.

## Using it

* UI: *Claims ledger* page - filter by paper/status/prior, open a claim, link experiments, set status.
* API: `GET /api/v1/claims`, `GET /api/v1/claims/{id}`, `PUT /api/v1/claims` (see `docs/API.md`).
* Python: `forge.planner.claims` (ledger CRUD in the sqlite db; the module lists the exact names).

## Honest scope

The ledger is bookkeeping: it does not verify anything by itself. At release time **all claims are `untested`
except those that a test in this repository reproduces with synthetic data** (for example formula-level facts such
as KV bytes per token or the E2M1 grid). A claim checked on synthetic tensors is not a claim about real models;
say so in `notes`.

## Ternary (tern158) numbers are not in the ledger yet

The tern158 report's claims (never learn the scale; one Lloyd step recovers ~86% of the absmean gap; equilibrium flip rate 0.59*sqrt(2*lr*wd); zero density 40-52% healthy; INT8 pipe ~1/30 of
FP8 on sm_103; 8x less traffic from gathering packed codes) and tern158's own sm_120 measurements (mma 346-411 TOPS, 125M run ppl 141 at step 500) are **not seeded** as ledger claims: the
seed tests tie every claim to a planner rule and no ternary rule exists yet. Until then they appear only as labelled TARGET values (health thresholds, gate `g-t158`, simulator caption) and in
`docs/TERNARY.md`; none of them has been reproduced through Forge.
