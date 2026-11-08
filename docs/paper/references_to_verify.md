# References requiring verification

No citation in the draft was invented. The table separates what the repository documents from what still needs a real literature search. No author lists, venues or DOIs should be added from memory. Verify each against the original source.

## A. Sources documented in the repository (details partly verified)

| # | Draft entry | What is verified | What still needs checking |
|---|---|---|---|
| [1] | J. Cheng et al., "Enhanced performance of brain tumor classification via tumor region augmentation and partition," PLoS ONE 10(10), 2015 | Title, journal, volume/issue and year, as stated in the dataset README (`data/raw/figshare_1512427/original/README 2024.txt`) | Full author list, article number, DOI |
| [2] | J. Cheng, "brain tumor dataset," figshare v8, doi:10.6084/m9.figshare.1512427, CC BY 4.0 | DOI, version, license and author, from the Figshare API record (`figshare_article.json`, `download_provenance.json`) | Preferred citation format |
| [3] | MaleCNS connectome, `male-cns:v1.0`, https://male-cns.janelia.org/, preprint doi:10.1101/2025.10.09.680999 | Dataset name, homepage, preprint DOI and CC-BY license (fetch metadata) | Authors, title, final venue (the homepage mentions a 2026 Cell paper) |
| [4] | neuPrint, https://neuprint.janelia.org; neuprint-python v0.6.3 | URL and software version | A citable neuPrint publication |
| [5] | FlyVis, https://github.com/TuragaLab/flyvis | Repository URL (from the project brief) | The associated publication. The config comment "Lappalainen et al. 2024" was written from memory and is **unverified** |
| [6] | Nematode Connectome Neural Network, https://github.com/LongJin-lab/Nematode-Connectome-Neural-Network | Repository URL (from the project brief) | The associated publication |
| [7] | J. Cheng et al., "Retrieval of brain tumors by adaptive spatial pooling and Fisher vector representation," PLoS ONE 11(6), 2016 | Title, journal and year, as stated in the dataset README | Full author list, article number, DOI |

## B. Unresolved placeholders in `paper_draft.md`

1. [CITATION NEEDED: published NCNN paper]. Section 1 and Section 2.
2. [CITATION NEEDED: FlyVis publication]. Section 1 and Section 2.
3. [CITATION NEEDED: Maslov & Sneppen, degree-preserving rewiring]. Section 1.
4. [CITATION NEEDED: ER, WS and BA original papers]. Section 2.
5. [CITATION NEEDED: randomly wired neural networks for image recognition]. Section 2.
6. [CITATION NEEDED: Takemura et al.; Shinomiya et al.; review of fly motion vision]. Section 2. These names also appear in `configs/malecns_visual.yaml` comments, written from memory and **unverified**.
7. [CITATION NEEDED: GNN survey or reference]. Section 2.
8. [CITATION NEEDED: representative CNN studies on this dataset]. Section 2.
9. [CITATION NEEDED: data leakage in medical imaging]. Section 2.
10. [VERIFY: literature search required before any novelty claim]. Section 1, the "To our knowledge, few studies…" sentence.

## C. Rule
If a placeholder cannot be filled with a verified source, rewrite or remove the sentence rather than keeping an unsupported claim.
