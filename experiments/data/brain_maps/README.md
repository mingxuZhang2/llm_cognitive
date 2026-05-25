# Brain Maps: Ground-Truth Reference Data

Brain imaging reference data for LLM-brain alignment analyses. All data are group-level / meta-analytic — no individual subject data are stored locally.

## Directory layout

```
brain_maps/
├── neurosynth/             # 27 meta-analytic activation maps (54 .nii.gz)
│   ├── *.nii.gz            # association maps (MKDA Chi-square vs all others)
│   ├── *_uniformity.nii.gz # uniformity test maps
│   ├── manifest.json       # full inventory + per-term study counts
│   └── raw/                # Neurosynth v7 source corpus (14,371 studies)
├── hcp/                    # 20 HCP S1200 task contrasts
│   ├── *.nii.gz            # group-average z-stat maps
│   └── manifest.json
├── MOFOMIC_INFO.md         # MOFOMIC dataset access notes
└── PUBLIC_DATA_SURVEY.md   # survey of other public fMRI datasets
```

## Neurosynth coverage

27 terms across 4 cognitive domains, all with ≥75 studies (well above the 30-study reliability threshold).

| Domain | Terms (n_studies) |
|---|---|
| **Emotion** | anger (89), disgust (103), fear (363), sad (163), happy (225), happy faces (75), emotion (1037), affective (748), valence (361), arousal (295), amygdala (1579) |
| **Moral** | moral (87), judgment (290), intention (95), intentional (100), intentions (125) |
| **ToM/Social** | theory mind (181), mentalizing (151), social cognition (220), social (1302), social interaction (117), belief (83), beliefs (77), empathy (187) |
| **Self** | self (1206), self referential (166), self report (77) |

**Method**: Multilevel Kernel Density Analysis (MKDA) Chi-square association test via NiMARE, run against the Neurosynth v7 corpus (14,371 fMRI studies, abstract-derived TF-IDF features, threshold 0.001, 10mm kernel).

**Two map types per term:**
- **Association test** (`{term}.nii.gz`): voxels where the term is preferentially reported (controls for base rate). USE THIS for selectivity claims.
- **Uniformity test** (`{term}_uniformity.nii.gz`): voxels consistently reported across studies. Useful as a sanity check.

## HCP contrast coverage

Group-average task fMRI z-stat maps from the HCP S1200 release (1096 subjects). Downloaded from NeuroVault collection 457 — these are the canonical Barch et al. 2013 contrasts available without ConnectomeDB registration.

| Task | Contrasts |
|---|---|
| **SOCIAL** | tom, random, tom_vs_random |
| **EMOTION** | faces, shapes, faces_vs_shapes |
| **LANGUAGE** | story, math, story_vs_math |
| **RELATIONAL** | rel, match, rel_vs_match |
| **WM (working memory)** | 0bk, 2bk, 2bk_face, 2bk_vs_0bk |
| **GAMBLING** | reward, punish, reward_vs_punish |

**Most useful contrasts for our paper:**
- `social_tom_vs_random.nii.gz` → ToM/mentalizing network ground truth
- `emotion_faces_vs_shapes.nii.gz` → emotion processing ground truth
- `language_story_vs_math.nii.gz` → language network ground truth
- `relational_rel_vs_match.nii.gz` → relational reasoning ground truth

## Recommended usage for LLM-brain alignment

For RSA-style alignment analyses, the workflow is:

1. **For each cognitive construct**, get its brain activation pattern from Neurosynth (1 map per term) or HCP (1 contrast).
2. **Sample voxels within a relevant atlas/ROI** (e.g., the Schaefer 400-parcel atlas, the Yeo 17-network atlas, or specific ROIs like vmPFC/TPJ/amygdala).
3. **Construct stimulus-by-stimulus RDMs** on the LLM side (using neuron module activations across our cognitive stimuli).
4. **Compare** via Mantel test / RSA / encoding model. Partial RSA controlling for surface-semantic similarity is recommended to avoid trivial alignment.

**Combination strategy for the paper:**
- Neurosynth provides BROAD coverage (any term) — good for breadth claims and for terms we don't have HCP equivalents for (moral, self-referential)
- HCP provides DEEP, high-quality SPECIFIC contrasts with matched control conditions (faces vs shapes is properly subtracted) — good for our key contrasts where HCP has them

## Reproducibility

Both manifest.json files contain full provenance for every file: source DOI, NeuroVault collection/image IDs, Neurosynth term, study count, download timestamp, MKDA parameters.

To regenerate: see the build scripts under `experiments/src/` (to be added in next phase).

## Provenance

- **Neurosynth v7**: https://neurosynth.org/ (Yarkoni et al. 2011 Nat Methods)
- **HCP S1200**: https://www.humanconnectome.org/ (Van Essen et al. 2013 NeuroImage; Barch et al. 2013 NeuroImage)
- **NeuroVault collection 457**: https://neurovault.org/collections/457/
- **NiMARE**: https://nimare.readthedocs.io/ (Salo et al. 2023 Aperture Neuro)

## Status notes

- 2026-05-25: All 27 Neurosynth maps + 20 HCP contrasts downloaded successfully
- Brain data agent crashed (API 529 overload) during final documentation write — data files are complete, only this README and MOFOMIC notes were added post-hoc
- MOFOMIC investigation pending — see MOFOMIC_INFO.md placeholder
