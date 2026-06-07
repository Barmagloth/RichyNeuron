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
