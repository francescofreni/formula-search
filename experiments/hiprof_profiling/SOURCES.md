# Search provenance

Every formula in `configs.py` was copied verbatim from a saved search output.
Graphs and targets also match the source; no variables were renamed and no
extra factors or quotient expansions were introduced. Full-search means a
bounded derivation run, not enumeration of every possible formula.

The adjustment graph is now the actual search benchmark:
`A -> B; B -> X; A -> Y; X -> Y`. Front-door retains treatment `T`,
as recorded in its source. Other groups target the effect of `X` on `Y`.

## Presets

Indices below are **zero-based** JSON array indices, not class labels.
Complexity counts probability terms (both sides of a quotient separately),
then sums, internal conditional divisions (`icd`), and quotients. Ties are
allowed. This is a structural ordering, not a measured runtime ranking.

| Preset | Source | JSON entry | Terms | Operators |
| --- | --- | --- | ---: | ---: |
| `ADJUSTMENT_01_ADJUST_A` | astar-adjustment | `classes[0]` | 2 | 1 |
| `ADJUSTMENT_02_NESTED_ADJUSTMENT` | full-adjustment | `formulas[1]` | 3 | 2 |
| `ADJUSTMENT_03_CONDITIONAL_DIVISION` | full-adjustment | `formulas[5]` | 3 | 3 |
| `ADJUSTMENT_04_TREATMENT_AVERAGING` | full-adjustment | `formulas[8]` | 4 | 2 |
| `ADJUSTMENT_05_NESTED_REWEIGHTING` | full-adjustment | `formulas[65]` | 6 | 4 |
| `MEDIATION_01_ADJUST_Z` | full-mediation | `formulas[0]` | 2 | 1 |
| `MEDIATION_02_MEDIATOR_FACTORIZATION` | full-mediation | `formulas[2]` | 3 | 1 |
| `MEDIATION_03_CONDITIONAL_DIVISION` | beam-mediation | `classes[5]` | 3 | 3 |
| `MEDIATION_04_SEPARATE_ADJUSTMENTS` | beam-mediation | `classes[4]` | 4 | 3 |
| `MEDIATION_05_NESTED_CONDITIONAL_DIVISIONS` | beam-mediation | `classes[23]` | 7 | 6 |
| `FRONT_DOOR_01_STANDARD` | full-front-door | `formulas[0]` | 3 | 2 |
| `NAPKIN_01_MARGINAL_WEIGHTS` | full-napkin | `formulas[0]` | 3 | 3 |
| `NAPKIN_02_CONDITIONAL_WEIGHTS` | full-napkin | `formulas[1]` | 3 | 3 |

## Saved files

Paths are relative to the repository root. Result files are Git-ignored;
the copied presets work without them. SHA-256 hashes identify the exact
source files used during curation.

- **astar-adjustment**: [experiments/tree_search/astar/results/fleet-20261023T122346Z-dw6r1F/runs/20261023T122350Z-case000-3b72f577/result.json](../../experiments/tree_search/astar/results/fleet-20261023T122346Z-dw6r1F/runs/20261023T122350Z-case000-3b72f577/result.json)
  SHA-256: `16c3594df14d1e005ce29941e51fda8c543fc8fcfd94f34ccc2736ad098066e9`.
- **full-adjustment**: [experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/adjustment.json](../../experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/adjustment.json)
  SHA-256: `1fede899888c8adc91bf30c32bfc6ba4f7fec4e8cff654a875ad3d71968c5418`.
- **full-mediation**: [experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/mediation.json](../../experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/mediation.json)
  SHA-256: `8e8ed49ae09b8b8fedba2117e31a6c3359e35a8f16af94f37b7bec509ef6762e`.
- **full-front-door**: [experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/front-door.json](../../experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/front-door.json)
  SHA-256: `a3e8263f65e715d1638fecd546dfd15e13fa9d952360b7d69367b802d055f961`.
- **full-napkin**: [experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/napkin.json](../../experiments/tree_search/derivation/results/four-tasks-20261023T131147Z/napkin.json)
  SHA-256: `d880e41a3210f030df5a2fcb80c1bcddc75229b893f57762426f515c95f468ce`.
- **beam-mediation**: [experiments/results/2026-10-31_192406Z__mediation-beam-prompts-five-repeats__99bd7b/mediation/beam-v3/repeat-03/result.json](../../experiments/results/2026-10-31_192406Z__mediation-beam-prompts-five-repeats__99bd7b/mediation/beam-v3/repeat-03/result.json)
  SHA-256: `49f9b7ff455ede7937cbf6afaeb6975c75e771197f4fd95ee3b26230e21d2eb1`.

The A* example used uniform-cost ordering with 200 pops. The full-search
files used bound 4. The beam examples used exploratory-v3, repeat 3, width
10, and 50 layers on the mediation task.

Front-door supplies one archived class and napkin two in the selected
bounded results; no synthetic variants were added to pad these groups.
Archival class membership and HiProF acceptance do not imply mathematical
novelty or unrestricted validity beyond the verifier's model family.
