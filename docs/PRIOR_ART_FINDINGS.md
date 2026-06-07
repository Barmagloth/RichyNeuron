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

## DECISIVE prior art found (deeper search, per user "search better")
**GateLORD — Zucchet & Orvieto, "Recurrent neural networks: vanishing and exploding
gradients are not the end of the story", arXiv 2405.21064 (NeurIPS 2024).**
- Computes the output as **`p(x_t, h_t) ⊙ o(x_t, h_t)`** — a MULTIPLICATIVE output where
  both branches are functions of input x_t AND the latent state h_t. This is exactly
  the axis-2 family: a STATE-DEPENDENT multiplicative output gate, on a DIAGONAL linear
  RNN (our regime).
- **They run our ablation:** `p⊙o` (with the multiplicative gate) vs `p` alone (without)
  — and find the multiplicative state-dependent output gate **substantially improves
  prediction**. I.e. the mechanism, the controlled ablation, AND a positive result are
  already published, on a diagonal linear recurrent model.
- Corroborating: HGRN-2 and Gated DeltaNet "expose their hidden state to a learned
  token-wise gate" and are noted as best-in-recall once hybridised — the importance of
  state-dependent gating is established.

## FINAL verdict: axis 2 is prior art (mechanism + ablation + result). FULL not warranted.
Our axis 2 (state-dependent multiplicative output gate over a diagonal state) and its
isolated contribution are NOT novel: present as LSTM (1997), xLSTM sLSTM (2024), and —
decisively — GateLORD (2024) already performs the equal-design ablation on a diagonal
linear RNN and reports that the state-dependent multiplicative gate helps. Running FULL
(A-fix-off) would re-derive a published result. Per PRIOR_ART_CHECK §5, this is a valid,
cheap, important outcome (saved ~20h CPU, established the novelty boundary). Recommend:
do NOT run FULL as framed; the project's remaining open direction (if any) must be
re-scoped against this prior art. Decision escalated to the user.

## Sources (deeper)
- GateLORD / RNN gradients: https://arxiv.org/abs/2405.21064
- xLSTM: https://arxiv.org/abs/2405.04517
- Griffin/Hawk: https://arxiv.org/abs/2402.19427
- Gated DeltaNet: https://arxiv.org/abs/2412.06464
- HGRN2: https://arxiv.org/abs/2404.07904
- GLA: https://arxiv.org/abs/2312.06635

## PHASE / COMPLEX channel branch (second prior-art sweep, per user)
Checked branches A-E (complex/phase recurrent, oscillatory SSM, amplitude-envelope,
phase-multiplexing, neuroscience phase coding) via search → fetch → snowball. The
verdict depends on which of TWO interpretations the phase channel means — and BOTH are
heavily covered:

**Interpretation 1 — complex/oscillation as a DYNAMICS/memory tool (like S4):**
- Unitary/orthogonal RNNs (Arjovsky 2016, arXiv 1611.00035): complex hidden state, norm-
  preserving, long memory, can implement DFT/spectral representations.
- **LinOSS — Oscillatory State-Space Models** (Rusch et al., arXiv 2410.03943, 2024/25):
  linear SSM as forced harmonic oscillators ("inspired by cortical dynamics"), parallel
  scan, SOTA long-range. Oscillation = stable long-memory device (not phase-as-channel).
- S4 (complex eigenvalues) and Mamba-3 (complex-valued state) — complexity as a memory
  trick. → **Interpretation 1 is hard-closed.**

**Interpretation 2 — phase carries SEPARATE information (rate+phase multiplexing / channel):**
- **Deep Complex Networks** (Trabelsi et al., 2017, arXiv 1705.09792): explicitly
  "amplitude = firing rate, phase = relative timing; similar phases add constructively
  (synchrony)". The rate+phase dual-coding idea, published 2017.
- **Theta-gamma phase code** (computational neuroscience): gamma = item content, theta-
  phase = sequence order; working-memory models exploit it for multi-item ordered
  sequences (the neuroscience origin of phase-multiplexing).
- **Phase-Associative Memory** (Vishwakarma & Agostino, arXiv 2604.05030, 2026): complex-
  valued sequence LM whose phase carries semantic info (distinct phase for synonyms vs
  unrelated) — but only "competitive with a real-valued ablation" (benefit modest/unclear).
- Audio: complex-spectrogram amplitude/phase separation (e.g. APSS, arXiv 2509.13825) —
  amplitude & phase as parallel estimated streams. → **Interpretation 2 is also covered**
  (ML 2017+2026, neuroscience, audio), with the ML param/quality benefit reported modest.

**Verdict (phase branch):** like axis 2, the phase/complex channel is prior art on BOTH
readings. As a dynamics tool: hard-closed (unitary RNN / LinOSS / S4 / Mamba-3). As a
semantic/timing channel: covered (Deep Complex Networks 2017; theta-gamma; PAM 2026), and
the reported benefit over real-valued is modest — so even the "does phase-channel buy
anything at equal params" ablation is partly foreknown (≈ "competitive, not clearly better").

## Sources (phase branch)
- LinOSS / Oscillatory SSM: https://arxiv.org/abs/2410.03943
- Unitary Evolution RNN (Arjovsky): https://arxiv.org/abs/1511.06464
- Deep Complex Networks: https://arxiv.org/abs/1705.09792
- Phase-Associative Memory: https://arxiv.org/abs/2604.05030
- Theta-gamma phase code (review): https://pubmed.ncbi.nlm.nih.gov/23522038/
