# FlyConnectomeMRI paper draft: LaTeX project for Overleaf

LaTeX version of `docs/paper/paper_draft.md`. The text was transferred mechanically, without scientific editing. Only the typesetting changed: Markdown markup became LaTeX commands, formulas are typeset in LaTeX, tables are LaTeX `tabular`, and figures are floats.

**Formatting:** IEEEtran in `conference` mode, as *preliminary* formatting. This is **not** the official ISBI 2027 template; the conference requirements still need to be checked.

## Contents

```
main.tex                  preamble, title, \input of every section, bibliography
references.bib            bibliography ([2] complete; all others are marked DRAFT ENTRY)
sections/00_abstract.tex
sections/01_introduction.tex
sections/02_related_work.tex
sections/03_methods.tex      (equations 1–9, Table II, Figs. 1–3)
sections/04_experiments.tex
sections/05_results.tex      (Tables S1, III, IV, V; Figs. 4–7)
sections/06_discussion.tex
sections/07_limitations.tex
sections/08_conclusion.tex
sections/09_reference_notes.tex   notes from the "References" section of the Markdown
sections/10_alternative_titles.tex
figures/*.png             the 7 figures used in the draft (unchanged copies)
```

All paths are relative. `IEEEtran.cls` and `IEEEtran.bst` are not included, because Overleaf and TeX Live already provide them.

## Upload to Overleaf

1. Overleaf → **New Project** → **Upload Project** → choose `FlyConnectomeMRI_Overleaf.zip`.
2. Menu → **Settings**: Compiler **pdfLaTeX**, Main document `main.tex`.
3. Click **Recompile**. Overleaf runs BibTeX automatically.

## Compile locally (if TeX Live or MiKTeX is installed)

```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Unresolved items

- **Placeholders:** `[CITATION NEEDED: …]` (red) and `[VERIFY: …]` (orange) are typeset with the `\citeneeded{}` and `\verifyneeded{}` macros. They stay visible in the PDF next to the text they belong to.
- **Draft references:** every entry in `references.bib` except [2] is a **DRAFT ENTRY**. Its `note` field keeps the unresolved marker. Do not replace these with guessed publications.
- **Authors:** the author block is a placeholder: `[Authors to be added]`.
- **Numbering:** table numbers II, S1, III, IV and V are kept as in the Markdown through `\settablenumber{}`. Reference numbers [1]–[7] are kept through `\nocite` in `main.tex`.
