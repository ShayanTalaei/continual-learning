---
noteId: "fb4576709dcf11f0b67f67467df34666"
tags: []

---

## Documentation Index

Comprehensive documentation for the continual learning framework.

### Getting Started

- **[Quickstart](quickstart.md)** - Get up and running in 5 minutes
- **[Main Overview](main.md)** - Framework goals and architecture
- **[Configuration Reference](reference/config.md)** - Complete YAML configuration guide

### Core Concepts

Understand the framework's building blocks:

- **[Agents](concepts/agents.md)** - Agent types, memory, and learning modes
  - HistoryAgent, MemorylessAgent, ReflexionAgent
  - Training/evaluation modes, agent cloning
- **[Memory](concepts/memory.md)** - Memory modules and update functions
  - HistoryList, snapshots, persistence
- **[Environments](concepts/environments.md)** - Task environments and datasets
  - QA, Math, MCQ, Finer, OMEGA Math, ALFWorld
  - Single-turn and multi-step tasks
- **[Runtime](concepts/runtime.md)** - Training loop, validation, scoring
  - Episode execution, parallel validation, checkpointing
- **[Language Models](concepts/language_model.md)** - LLM clients and configuration
  - Gemini, OpenAI, Tokasaurus (local inference)
  - Retry logic, call logging
- **[Checkpointing](concepts/checkpointing.md)** - Save and resume training
  - Checkpoint strategies (last-N, top-K validation)
  - Memory snapshots, resume workflow
- **[Logging](concepts/logging.md)** - Structured logging system

### Practical Guides

Step-by-step tutorials:

- **[Checkpointing Guide](guides/checkpointing.md)** - Save progress and resume training
- **[Validation Guide](guides/validation.md)** - Monitor performance on held-out data
- **[Multi-Step Environments](guides/multi-step-environments.md)** - Interactive tasks (ALFWorld)
- **[Tokasaurus Setup](guides/tokasaurus-setup.md)** - Local inference with open-source models
- **[Running Benchmarks](guides/running-benchmarks.md)** - Standard benchmark integration
- **[Synthetic Data](guides/synthetic-data.md)** - Generate synthetic datasets

### Cookbooks

Complete example configurations:

- **[Encryption with History](cookbooks/encryption-history-list.md)** - Basic HistoryAgent setup

### Design Documents

Architecture and planning:

- **[Roadmap](design/roadmap.md)** - Future features and priorities
- **[Plan](plan.md)** - Detailed implementation notes and status
- **[Short/Long Memory Design](design/short-long-memory-design.md)** - Memory architecture

---

## Quick Navigation by Use Case

**I want to...**

- **Start my first experiment** → [Quickstart](quickstart.md)
- **Understand agents** → [Agents Concept](concepts/agents.md)
- **Use local models** → [Tokasaurus Setup](guides/tokasaurus-setup.md)
- **Save and resume training** → [Checkpointing Guide](guides/checkpointing.md)
- **Monitor validation** → [Validation Guide](guides/validation.md)
- **Configure everything** → [Config Reference](reference/config.md)
- **Work with multi-step tasks** → [Multi-Step Environments](guides/multi-step-environments.md)
- **See what's implemented** → [Plan](plan.md)
- **Understand the vision** → [Main Overview](main.md)

---

## Documentation Structure

```
docs/
├── README.md              # This file
├── quickstart.md          # Quick start guide
├── main.md                # Framework overview
├── plan.md                # Detailed implementation notes
├── concepts/              # Core concept documentation
│   ├── agents.md
│   ├── memory.md
│   ├── environments.md
│   ├── runtime.md
│   ├── language_model.md
│   ├── checkpointing.md
│   └── logging.md
├── guides/                # Practical how-to guides
│   ├── checkpointing.md
│   ├── validation.md
│   ├── multi-step-environments.md
│   ├── tokasaurus-setup.md
│   ├── running-benchmarks.md
│   └── synthetic-data.md
├── reference/             # API and config reference
│   └── config.md
├── cookbooks/             # Example configurations
│   └── encryption-history-list.md
└── design/                # Design documents
    ├── roadmap.md
    └── short-long-memory-design.md
```


