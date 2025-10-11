---
noteId: "37233ec09cd111f08a082794fdda1c76"
tags: []

---

{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # STM \uc0\u8596  LTM Memory for LLMs with a KV \'93Cartridge\'94\
\
A clean specification you can paste into a repo or design doc. It formalizes a two-part memory for LLMs:\
1) **Short-term memory (STM):** natural-language items appended to the KV cache (context).\
2) **Long-term memory (LTM):** an **optimized prefix** (\'93cartridge\'94) learned from accumulated STM.\
\
The design exploits complementary strengths:\
- STM is **sample-efficient and instant** (appendable per item) but bounded by window/latency/recall degradation.\
- LTM is **compact, fast at inference, and generalizes** but requires **batch evidence** to train safely.\
\
We define measurable signals, decision rules, and a switching inequality to convert STM into an LTM update at the right time.\
\
---\
\
## Contents\
- [Notation](#notation)\
- [Goals & Design Principles](#goals--design-principles)\
- [Key Metrics](#key-metrics)\
- [Switching Rule (STM \uc0\u8594  LTM)](#switching-rule-stm--ltm)\
- [Single Inequality (Actionable Form)](#single-inequality-actionable-form)\
- [Thresholds & Sizing](#thresholds--sizing)\
- [Implementation Pattern (Streaming)](#implementation-pattern-streaming)\
- [Heuristics & Defaults](#heuristics--defaults)\
- [Evaluation Protocol](#evaluation-protocol)\
- [Risks & Mitigations](#risks--mitigations)\
- [Extensions](#extensions)\
\
---\
\
## Notation\
\
- \\( S_t=\\\{(x_i,y_i)\\\}_\{i=1\}^\{n_t\} \\) \'97 STM buffer (the items you appended so far).\
- \\( \\phi_t \\) \'97 parameters of the LTM **cartridge** (e.g., learned prefix/LoRA).\
- \\( N_\{\\max\} \\) \'97 effective STM capacity (beyond which recall/latency degrades).\
- \\( b \\) \'97 average token cost per STM item (including separators/markup).\
- \\( Q \\) \'97 expected number of **future queries** that will benefit before the next consolidation.\
- \\( \\mathcal\{L\}(\\cdot) \\) \'97 task loss (e.g., NLL or task-specific metric).\
- \\( \\hat\{L\}_\{\\text\{ctx\}\}(n) \\) \'97 loss using STM of size \\(n\\).\
- \\( \\hat\{L\}_\{\\text\{cart\}\}(S_t) \\) \'97 predicted loss **after** training a cartridge on \\(S_t\\) (estimated via a fast probe).\
- \\( c_\{\\text\{ctx\}\}(n) \\) \'97 latency/compute cost per query with STM size \\(n\\).\
- \\( r_\{\\text\{cart\}\} \\) \'97 per-query overhead of using the cartridge (usually small).\
- \\( c_\{\\text\{upd\}\} \\) \'97 one-off cost to train/update the cartridge.\
- \\( q_\{\\text\{ctx\}\}(n) \\) \'97 STM recall probability (right snippet used) given \\(n\\) items.\
- \\( \\text\{SNR\}(S_t)=\\frac\{\\|\\mathbb\{E\}[g]\\|\}\{\\sqrt\{\\operatorname\{Tr\}\\operatorname\{Var\}(g)\}\} \\) \'97 gradient signal-to-noise from a probe fit on \\(S_t\\) (empirical Fisher proxy).\
- \\( DL(\\Delta\\phi) \\) \'97 description length (bits) of the **delta** update to the cartridge (MDL proxy).\
- \\( D_\{\\text\{resid\}\} \\) \'97 bits left in STM after consolidation (kept residuals).\
- \\( \\gamma \\) \'97 required quality margin (how much better LTM should be than STM).\
- \\( \\tau \\) \'97 SNR threshold for safe updates.\
\
---\
\
## Goals & Design Principles\
\
1. **Immediate utility:** STM should accept new items and improve results *now*.\
2. **Compiled knowledge:** Promote stable knowledge from STM \uc0\u8594  LTM when it\'92s cheaper and more accurate to keep as a compact prefix.\
3. **Evidence-based updates:** Update the cartridge only when we have adequate, stable signal (sample size & SNR).\
4. **Amortized efficiency:** The LTM update pays for itself across expected future queries \\(Q\\).\
5. **Compression wins:** Prefer the representation (STM tokens vs. cartridge params) with smaller description length for equal or better loss.\
\
---\
\
## Key Metrics\
\
### Quality\
- **STM recall curve:** \\( q_\{\\text\{ctx\}\}(n) \\) vs. \\(n\\).\
- **Loss curves:** \\( \\hat\{L\}_\{\\text\{ctx\}\}(n) \\) and probe-predicted \\( \\hat\{L\}_\{\\text\{cart\}\}(S_t) \\).\
- **Generalization check:** hold-out within \\(S_t\\) to ensure LTM isn\'92t memorizing.\
\
### Cost & Operations\
- **Latency:** \\( c_\{\\text\{ctx\}\}(n) \\) growth; p95 vs. SLO.\
- **Update cost:** \\( c_\{\\text\{upd\}\} \\), plus per-query \\( r_\{\\text\{cart\}\} \\).\
- **Compression:** \\( G_\{\\text\{MDL\}\} = n_t b - DL(\\Delta\\phi) - D_\{\\text\{resid\}\} \\).\
\
### Learning Signal\
- **SNR:** \\( \\text\{SNR\}(S_t) \\ge \\tau \\).\
- **Sample sufficiency:** approximate \\( m^\\star \\) needed for a meaningful update:\
  \\[\
  m^\\star \\approx \\frac\{DL(\\Delta\\phi) + \\log(1/\\delta)\}\{\\varepsilon\}.\
  \\]\
\
### Governance / Safety\
- **Contradictions/drift:** conflicts within \\(S_t\\).\
- **Regression guardrails:** protected eval suite (mu\
}