# PILOT 2 — analysis rubric (read BEFORE interpreting results)

Anti-optimism guardrails, fixed *before* seeing numbers. Apply all three when the
run finishes; do not let a favourable-looking number bypass them.

## 1. "Gap collapsed" is a SUCCESS of pilot 2, not a disappointment.
If, on a converged budget, B1 catches up to rich, pilot 2 did its job: it showed
pilot 1's gap was an under-training artifact, and it saved us a false FULL run
with inflated expectations. Hold "gap collapsed" as an **equally likely** outcome,
not a failure. Guard against the temptation to read noise as a surviving signal.

## 2. steps_to_best is now LOADED — interpret via the curve, not the number.
Under early-stop, steps_to_best mixes two things: real convergence speed AND the
moment patience happened to fire. Two models with the same real speed but
different plateau shape (smooth vs saw-toothed) get different steps_to_best purely
from min_delta dynamics, not from a compute advantage. So if rich "converges
faster", **verify it on the shape of the val curve**, not on the stop point alone.
A speed advantage is real only if visible in the val trajectory itself.

## 3. A1 Δ on the converged model may be weaker — read it WITH the ablated level.
Pilot 1 had Δ=0.47 on an under-trained model. On a trained model Δ may move either
way: trained rich may (a) lean on state more (Δ grows) or (b) partially compensate
for W_h=0 via better-learned other paths (Δ shrinks). The decisive question is the
**absolute level the ablated version holds**: if ablated stays at chance (~0.125),
the state is still load-bearing regardless of Δ. Only conclude "state is fictitious"
if ablated remains near the non-ablated accuracy.

## Standing reminders (carried from PROBLEM/SPEC)
- Pilot 2 is reconnaissance: NO H0 verdict, PREREG stays unfrozen.
- Metric is `test_at_best` (test at the best-VAL step); lr chosen by median VAL.
- Report the unfavourable reading if the numbers are ambiguous.
