# PRIOR_ART_FINDINGS.md — novelty verdict before FULL (v0.5.0)

> Code-read + light literature search per PRIOR_ART_CHECK.md. Verifiable record so
> the verdict is checkable, not "I looked, seems no". No CPU burned.

## What was read (state-spaces/mamba, raw files, June 2026)
- `mamba_ssm/modules/mamba_simple.py` (Mamba-1)
- `mamba_ssm/modules/mamba2.py` (Mamba-2)
- `mamba_ssm/modules/mamba3.py` (Mamba-3; confirmed real, arXiv 2603.15569, 2026-03-16)

## Gating mechanics extracted (the question: does the STATE enter the output-gate argument?)
- **Mamba-1:** `x, z = in_proj(u).chunk(2)`; gate `z` is **input-only**. State reaches
  output only via linear readout `C·state` inside `selective_scan_fn`, then
  `y = y * act(z)` (gate is input-only). State is NOT in the gate argument.
- **Mamba-2:** `z` from the `in_proj` split (input-only). State → output via
  `y = einsum("bhpn,bn->bhp", ssm_state, C)` (linear), then `y = norm(y, z)` /
  `y = y*act(z)` with input-only `z`. State NOT in the gate argument.
- **Mamba-3:** `z` from the split (input-only); state via linear `Q=C` readout, then
  `y = norm(y, z)`. Adds complex-valued state + MIMO, but the output gate `z` is
  still input-only. State NOT in the gate argument.

**Conclusion (code-verified):** our **axis 2** — `g = σ(W_g·x + W_h·s)`, `y = (W_v·x)·g`,
i.e. the recurrent STATE inside the argument of a multiplicative output gate that
multiplies a separate input value-path — is **absent from Mamba-1/2/3**. Mamba
deliberately keeps gates input-only (parallelizability); the state only ever reaches
the output linearly (`C·s`), then is gated by an input-only `z`.

## BUT — broader prior art (the honest caveat)
"A multiplicative gate whose argument includes the recurrent state" is the **classic
gated-RNN mechanism**: LSTM output gate `o_t = σ(W_o x_t + U_o h_{t-1})`, `h_t = o_t ⊙
tanh(c_t)`; GRU reset/update gates likewise depend on the state. So axis 2's essential
ingredient is **known in the LSTM/GRU family** — it is novel *relative to the Mamba
line* (which dropped it), not a brand-new mechanism.

Literature search ("state-dependent output gating SSM ablation", Mamba-vs-LSTM gating):
sources confirm Mamba "lacks the sophisticated gating structures found in LSTM/GRU"
and "rely instead on a single linear transformation" — i.e. the gap is acknowledged —
but **no prior controlled equal-params ablation isolating the contribution of
state-dependent output gating over a stripped SSM core was found.**

## Verdict (§3): AMBIGUOUS / known-elsewhere → escalate (per doc, not ours to decide)
- Axis 2 is NOT in any Mamba generation (verified).
- Axis 2's core idea IS the LSTM/GRU state-dependent gate (known mechanism Mamba dropped).
- The specific controlled ablation we would run (does re-introducing state-dependent
  gating into a diagonal selective-SSM buy parameters at equal quality) has **no found
  prior art** → methodological value even though the mechanism is not brand-new.

## Sources
- Mamba-3: https://arxiv.org/abs/2603.15569
- Mamba (selective SSM) overview: https://athekunal.medium.com/mamba-and-state-space-models-explained-b1bf3cb3bb77
- Mamba dives: https://www.oxen.ai/blog/mamba-linear-time-sequence-modeling-with-selective-state-spaces-arxiv-dives

## DEEPER targeted check (per user: xLSTM / Griffin-Hawk / Gated DeltaNet)
Shallow "not found" was insufficient. Checked the most likely candidates directly:

- **xLSTM (Beck et al. 2024, arXiv 2405.04517) — the decisive one.**
  - **sLSTM gates ARE state-dependent:** `o_t = σ(W_o·x_t + r_o·h_{t-1} + b_o)` (also
    i_t, f_t carry recurrent `r·h_{t-1}` terms). The OUTPUT gate depends on the hidden
    state — this IS our axis 2 mechanism. sLSTM is not parallelizable (memory mixing).
  - **mLSTM gates are input-only** (no hidden-state interaction) for parallelizability.
  - So xLSTM literally instantiates BOTH "state-dependent gate" (sLSTM) and "input-only
    gate" (mLSTM) — but as two DIFFERENT blocks (scalar vs matrix memory, mixing vs
    associative memory). Their comparison is CONFOUNDED, not an isolated equal-params
    toggle of state-in-gate. Their ablations cover exponential gating / norm / residual /
    memory-mixing — not an isolated state- vs input-gate contribution.
- **Griffin/Hawk (RG-LRU, DeepMind 2024):** input-dependent gates, **explicitly removes
  hidden-state dependence in gating** for parallelizability. No axis 2.
- **Gated DeltaNet (2024):** input-dependent erase/decay gating + delta rule; ablation
  isolates erase-gate vs delta-rule, not state-dependent output gating. No axis 2.

## Refined verdict
1. **Mechanism (state in the output-gate argument) is NOT novel** — classic LSTM (1997)
   and explicitly present in xLSTM **sLSTM** (`o_t=σ(Wx+r·h+b)`, 2024).
2. **The modern parallelizable-recurrent field deliberately dropped it** (Mamba 1/2/3,
   Griffin/Hawk, xLSTM-mLSTM all use input-only gates) — the state-vs-input gate tradeoff
   is implicitly known and is essentially the sLSTM-vs-mLSTM contrast.
3. **A CLEAN isolated equal-params ablation** ("toggle only state-in-gate over a minimal
   fixed-alpha diagonal SSM, all else fixed") was **not found** — but this is a deeper-
   yet-not-exhaustive search, and the qualitative answer is heavily implied by the field.
   Marginal novelty of our ablation is THIN; the likely outcome is partly foreknown.

Decision (whether the thin remaining methodological gap justifies the FULL run, reframe,
or pivot) is escalated to the user — not decided here.
