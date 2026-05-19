# Nature Paper Project: LLM Functional Atlas

## Project Goal
Build the first "Brodmann Atlas" of Large Language Models -- a comprehensive functional map validated by neuroscience gold standards (double dissociation), shown to be universal across architectures (Transformer, Mamba, RWKV), with quantitative brain alignment and developmental trajectory. Targeting Nature/NMI.

## Repository Structure
- `IDEA_SYNTHESIS.md` -- Main paper idea synthesis with 6-act story arc
- `research_paper_analysis.md` -- Analysis of ULCMOD and related prior work
- `research_neuroscience_methods.md` -- Survey of neuroscience methods applicable to LLMs
- `research_neuro_llm_landscape.md` -- Landscape survey of neuro-LLM research
- `research_nature_storytelling.md` -- Analysis of Nature-level storytelling patterns
- `feasibility_critical_periods.md` -- Feasibility study on critical periods angle
- `feasibility_convergent_evolution.md` -- Feasibility study on convergent functional organization across architectures (Transformer vs Mamba vs RWKV)

## Key Findings So Far
1. The specific "convergent evolution of functional modules across different architectures" angle has NOT been done as a unified study
2. Individual pieces exist: SAE feature universality (Towards Universality, ICLR 2025/2026), brain alignment across architectures (NeurIPS 2025 spotlight), functional modules in Transformers (Nature Comms 2024)
3. RWKV interpretability is nearly unexplored -- major opportunity
4. Working at the SAE-feature level on the residual stream is the methodologically safest path for cross-architecture comparison
5. The gap at the intersection of functional modules + multiple architectures + causal validation + brain alignment is wide open

## Current Status
- Phase: Literature review and feasibility assessment
- Branch: master

## Module Transplantation Feasibility (May 19, 2026)
- `feasibility_module_transplant.md` -- Exhaustive literature search on module/neuron transplantation between LLMs (50+ papers reviewed)
- Key finding: Neuron-level transplantation exists for safety functions (CNT, 2026) and agent roles (ARM, 2026), but nobody has transplanted causally-validated cognitive functional modules (math, code, reasoning) between different LLM architectures
- The field is moving fast -- multiple groups converging on transplantation ideas -- 12-18 month window of opportunity
- Closest papers: CNT (safety neurons), ARM (role-conditioned neurons), Beyond Learning (layer-level transplant), NOT (checkpoint-based layer blocks)
- Technical feasibility: POSITIVE -- all building blocks exist (module identification, neuron correspondence via OT, weight transfer), integration is the novelty
