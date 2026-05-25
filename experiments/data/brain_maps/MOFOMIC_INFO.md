# MOFOMIC Dataset Investigation

**Status**: Investigation deferred. Pilot experiment will use Neurosynth + HCP first.

## What MOFOMIC is

MOFOMIC (Moral Foundations fMRI Multi-study Cohort) is a 2026 fMRI dataset combining 4 studies, N=152 healthy adults, using Moral Foundations Vignettes (Clifford 2015) and socio-moral images, with graded moral wrongness ratings.

If accessible, it would be the ideal alignment target for our moral compositional finding, since:
- Same MFV stimuli we already have in `cognitive_stimuli/moral/moral_foundations_vignettes.jsonl`
- Per-stimulus brain activation patterns (not just group averages)
- Behavioral ratings + neuroimaging in same subjects

## To-do (when needed)

- [ ] Find the published 2026 MOFOMIC paper (likely Scientific Data or NeuroImage)
- [ ] Check OpenNeuro for the BIDS-formatted release
- [ ] Verify data use agreement requirements
- [ ] Estimate download size (preprocessed BOLD for 152 subjects ≈ 50-200 GB)
- [ ] Decide if individual-subject analyses are feasible or if group-average maps suffice

## Fallback

If MOFOMIC is unavailable or too large to handle, our HCP SOCIAL+EMOTION contrasts + Neurosynth moral/intention/judgment maps provide adequate ground truth for the alignment claims in the paper. MOFOMIC would strengthen but is not strictly required.
