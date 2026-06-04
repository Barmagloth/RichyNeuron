# RichyNeuron — the "rich unit" prototype

Can returning some computational complexity **into the unit** (temporal state that
modulates the unit's own multiplicative gate) make the architecture **more optimal**
— same quality with fewer parameters — than a split *temporal-mixing + channel-mixing*
stack, while staying on the mass-market stack (PyTorch, plain backprop, CPU)?

This is an **honest attempt to falsify the null hypothesis**, not to confirm a bet:

> **H0 (default, against us):** the rich unit does **not** reach the target quality
> with fewer parameters than the split stack, by a pre-registered margin `DELTA_MIN`.

H0 holds until refuted by the §5.4 rule. *"H0 not refuted, here are the numbers"* is
a full, publishable success of the prototype — not a failure. See `docs/` first.

## Read these first (in order)
1. `docs/RESEARCH_MAP_v0_2_0.md` — **§0 anti-bias rules, read before any work.**
2. `docs/PROBLEM_v0_2_0.md` — motivation and the self-deception traps.
3. `docs/SPEC_v0_2_0.md` — **frozen**, executable spec. The source of truth.

## Repository layout
```
.
├── docs/                         # frozen planning docs (PROBLEM / SPEC / RESEARCH_MAP)
├── rich_unit/                    # the package (SPEC §7)
│   ├── PREREG.md                 # §5.0 pre-registration — FREEZE before first sweep
│   ├── models/
│   │   ├── wrapper_v0_2_0.py     # shared embed → [core] → head scaffold (§4)
│   │   ├── rich_unit_v0_2_0.py   # variant B: RichUnitLayer (the unit under test, §2)
│   │   └── baselines_v0_2_0.py   # B1 (stacked) + B2 (GRU ref) (§4)
│   ├── tasks/
│   │   ├── selective_copy_v0_2_0.py   # §3.1
│   │   └── assoc_recall_v0_2_0.py     # §3.2
│   ├── train_v0_2_0.py           # single model-agnostic train loop (§4/§5)
│   ├── sweep_v0_2_0.py           # §5 protocol: minimal-width search
│   ├── report_v0_2_0.py          # §5.3/§5.4 table + verdict, §6 ablations
│   └── tests/                    # shapes, recurrence-vs-reference, generator determinism
├── results/                      # sweep outputs & reports (gitignored, dir kept)
├── ROADMAP.md                    # evolution path B → A → C → B+C+A (RESEARCH_MAP)
├── requirements.txt
└── pyproject.toml
```

> **Current state: STRUCTURE SCAFFOLD.** All `rich_unit/*` modules are documented
> **stubs** (`NotImplementedError`); tests are present but `skip`-marked. The
> implementation is the builder phase and starts **only after `PREREG.md` is frozen
> and committed** (SPEC §5.0). File naming follows the mandatory `vX_Y_Z` rule; bump
> the version on the first logic change rather than editing a frozen file in place.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Python ≥ 3.10
pip install -r requirements.txt
```
CPU-only by design (SPEC §1) — no CUDA, no triton/mamba-ssm. CUDA is used if present
but is never required or tested.

## Intended workflow (once implemented)
```bash
# 0. Freeze rich_unit/PREREG.md (ACC_TARGET, DELTA_MIN, N_SEEDS, aggregation) and commit.
pytest                              # shapes, recurrence vs hand ref, generator determinism
python -m rich_unit.sweep_v0_2_0    # §5: calibrate ACC_TARGET on B1, then min-width search
python -m rich_unit.report_v0_2_0   # §5.3/§5.4 table + verdict, A1/A2/A3 ablations
```

## Reconnaissance pilot (done)
A pilot (Selective Copy only, 3 seeds, NO H0 verdict) lives in
`rich_unit/pilot_v0_2_0.py`; results and the writeup are in
`results/PILOT_REPORT_v0_2_0.md`. Run: `python -m rich_unit.pilot_v0_2_0`.

## Long-run resilience (`rich_unit/runner_v0_2_0.py`)
The container is ephemeral, so long sweeps checkpoint to git:
- **Resume** — results are appended to a CSV; a restart skips cells already
  present (keyed by model × d_model × d_state × seed), so a reclaim costs only
  the unfinished tail.
- **Additive push** — every `CHECKPOINT_EVERY` cells the CSV is committed and
  pushed, force-added past `.gitignore` (commits are verified server-side on
  push). Disable with `RICH_UNIT_CHECKPOINT_PUSH=0` (used in tests / dry runs).

## How to read the verdict (SPEC §5.4)
- **H0 refuted on a task:** `params(RichUnit) ≤ (1 − DELTA_MIN) × params(B1)` with both at `ACC_TARGET` (median rule).
- **H1 fully supported:** refuted on **both** tasks. **Partial:** one task — weak signal, needs replication.
- **H0 not refuted** (incl. win `< DELTA_MIN`): a valid negative result.
- A failed **A1/A2** ablation **annuls** any formal win (profit without a working feature is just extra parameters).
