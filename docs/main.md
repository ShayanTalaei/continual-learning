---
noteId: "ed8f70109cf011f0b67f67467df34666"
tags: []

---

# Continual Learning

## Introduction and Goal
We study **continual learning**: an agent that receives observations, outputs actions, receives feedback, and updates its memory before proceeding. At step *t*:  

```math
M_t = U(M_{t-1}, o_t, a_t, f_t)
```

where $M_t$ is memory, $U$ is the update function, $o_t$ is the observation, $a_t$ is the action, and $f_t$ is feedback.  

**Goal:** Build agents that learn better from experience and improve performance over time. In particular, we want agents that can **learn at runtime** (“on the fly”) by updating their memory, not just at training.  

---

## General Framework
We separate two phases:  

- **Design time:** Choosing memory mechanisms and update functions, training parametric parts, and benchmarking.  
- **Run time:** The agent interacts with tasks, updates memory via its update mechanism, and adapts quickly to new data.  

This framework allows us to (1) compare different memory instantiations and update rules, and (2) eventually design learnable update functions that optimize adaptation during run time.  

---

# Memory Mechanisms and Update Functions (Detailed)

## 1. History of Experiences
**Memory:**  
A simple natural language text that contains all the experiences (observations, actions, feedback) seen so far.  

**Update Function:**  
Appending each new experience to the history.  

**Variants:**  
a) **Reflexion-style reflection:** Instead of raw appends, the agent can write reflections about the feedback, capturing *what worked*, *what failed*, and *what to try next*. This introduces a form of in-context reinforcement learning, where the model leverages past mistakes for better future actions.  

b) **Truncated history:** Maintain a fixed budget (e.g., last $k$ entries). Older experiences are truncated or summarized to prevent overflow while still retaining recency.  

---

## 2. Database of Experiences with Top-k Retrieval
**Memory:**  
A structured database of past experiences, where each entry corresponds to an (observation, action, feedback) tuple or a processed embedding of it.  

**Update Function:**  
Each new experience is added as a database entry.  

**Retrieval:**  
Given a new observation $o_t$, retrieve the top-$k$ most similar experiences using similarity search (e.g., cosine similarity in embedding space). These retrieved examples are inserted into the agent’s context for decision-making.  

**Pros/Cons:**  
- Pros: Scales better than raw history, retrieval targets relevant prior knowledge.  
- Cons: Large database size may become costly; retrieval errors can mislead.  

---

## 3. Dynamic Cheatsheet
**Memory:**  
A compact natural language text representing distilled learnings from past experiences—rules, strategies, “dos and don’ts.”  

**Update Function:**  
After each interaction, the agent integrates new feedback into the cheatsheet. This can take the form of rewriting existing rules, adding new ones, or removing obsolete advice.  

**Key Feature:**  
Unlike raw history or retrieval, the cheatsheet emphasizes *abstraction*—it compresses multiple experiences into generalizable patterns that improve sample efficiency.  

---

## 4. Cartridge
**Memory:**  
A *prefix* of the agent’s key–value (KV) cache in latent space. This prefix acts as a latent memory “cartridge” that augments the prompt at each step. Unlike text-based memory, cartridges operate directly in representation space.  

**Update Function (Prefix Tuning):**  
The cartridge is updated using gradient descent on a KL divergence objective.  

Formally, suppose the agent has a policy $\pi_\theta(a \mid o, C)$, where $o$ is the observation, $a$ the action, and $C$ the cartridge state. Given a set of experience samples $\mathcal{D} = \{(o_i, a_i)\}$, we update the cartridge by minimizing:

```math
\mathcal{L}_{\text{cartridge}}(C) = \frac{1}{|\mathcal{D}|} \sum_{(o_i, a_i) \in \mathcal{D}} D_{\text{KL}}\big( \pi_\theta(\cdot \mid o_i, C) \; \| \; \pi_\theta(\cdot \mid o_i, \mathcal{H}) \big)
```

Here:  
- $\pi_\theta(\cdot \mid o_i, \mathcal{H})$ is the agent’s distribution over actions conditioned on the raw history $\mathcal{H}$.  
- The objective encourages the cartridge $C$ to approximate the effect of conditioning on the full set of experiences, but in compressed latent form.  

---

### Cartridge Variants: Hybrid with Buffered History
We can combine cartridges with a **buffer of raw history**:  
1. Keep appending experiences to the history.  
2. Once the buffer reaches a threshold, flush its contents into the cartridge via prefix tuning.  
3. Reset or truncate the buffer and continue.  

**Benefits:**  
- Prevents unbounded history growth.  
- Relaxes the need to update the cartridge at every step (saves compute).  
- Provides sufficient batch samples for stable optimization.  

---

### Open Questions for Cartridges
**Optimization pairs:**  
On which (input, output) pairs should the cartridge be optimized?  

**Idea 1:** Use environmental feedback to create supervised pairs:  
- Input: observation $o_t$.  
- Target: action $a^*_t$ (the correct or improved action, derived from feedback).  

We can form training mini-batches by sampling from past experience:  
- For each flush, select a subset (e.g., 10%) of experiences.  
- Perform self-study with k-fold resampling to increase robustness.  

This produces consistent (observation, action) training signals to align the cartridge with successful strategies.  

---
 

---

## Experiments

### Benchmarks
We adapt standard benchmarks into **sequential settings**:  
- Construct $n$ sequences of $n$ tasks (circular permutations).  
- Agent solves tasks sequentially, receiving feedback before moving on.  
- Accuracy at step $i$ = average correctness of the $i$-th task across all sequences.  

To keep experiments tractable:  
- Group tasks into $p$ clusters (e.g., $p=5$) and permute groups instead of individual tasks.  
- Repeat with $k$ random permutations to avoid correlation.  

### Example Domains
- **AIME problems (math)**  
- **Text-to-SQL tasks**  
- **Embodied environments (e.g., Tales)**  

---

## Next Steps
1. **Phase 1:** Compare different memory architectures and update mechanisms (database, cheatsheet, cartridge).  
2. **Phase 2:** Develop **parametric update functions** that can be learned at design time and tested at run time.  
3. **Evaluation criteria:**  
   - Adaptation speed  
   - Memory size and growth  
   - Sample efficiency  
   - Long-term performance  
