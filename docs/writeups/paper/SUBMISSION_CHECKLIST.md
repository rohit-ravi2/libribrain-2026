# Submission checklist: leakage / evaluation paper

Prepared 2026-11-02, updated 16:30 EST. Nothing here has been submitted, posted or emailed. Rohit posts.
Sources: bioRxiv FAQ (biorxiv.org/about/FAQ) and arXiv endorsement help (info.arxiv.org/help/endorsement.html), both
fetched 2026-11-02. Items marked DECIDE are Rohit's call.

## Before either venue

- [x] Final numbers in: seeded CIs (job 2), holdout-format timing and timing + MEG (job 3). PDF rebuilt 2026-11-02
      (10 pages). Still to do by Rohit: read the PDF end to end.
- [x] Every number in the PDF traces to a file named in Appendix A or in the text (independent audit 2026-11-02, ~170
      numbers checked; mismatches fixed in 15b6eaf). Remaining soft source: "0.64 to 0.69 on new subjects" (4.1) spans
      two logged results rather than one file.
- [x] References: all 13 checked by fetch (`docs/writeups/references_verified.md`).
- [ ] DECIDE: courtesy note to the PNPL organisers before posting. The paper reports a non-exchangeable public/private
      split and the cross-subject channel in their benchmark; they hold the row-level split that would explain
      Section 4.7, and the competition paper "discourage[s] any attempt to infer held-out labels outside of the official
      evaluation procedure." A short note with the PDF invites the split data and avoids a surprise.
- [ ] DECIDE: the consensus-label holdout analysis (`private/holdout_consensus_timing.HELD_FOR_ROHIT.py`, not run to
      completion, no outputs kept). It would measure timing leakage on the real holdout sentences, but it infers
      held-out labels across subjects. Default: leave it out.
- [ ] DECIDE: whether to correct "8th place" on the repo description / site / blog to "8th public, 17th private".
- [ ] Repo: the paper cites github.com/rohit-ravi2/libribrain-2026; the scripts it names must be pushed before the
      preprint goes up (Rohit pushes).
- [ ] Author line and affiliation: "Rohit Ravi, New York University". Add ORCID if you have one. Add the NYU email
      as the contact address.

## bioRxiv (no endorsement needed)

- [ ] Article type: New Results. Category: Neuroscience (the only fit; computational papers with full methods and
      results are in scope per the FAQ).
- [ ] Upload: one PDF (`docs/writeups/paper/leakage_paper.pdf`); LaTeX must be converted first, which build.py does.
- [ ] License: choose one of CC0, CC-BY, CC-BY-NC, CC-BY-ND, CC-BY-NC-ND or no reuse. Suggest CC-BY (most reusable).
      The LibriBrain data are CC-BY-NC-4.0 but are not redistributed in the paper, so they do not constrain the
      paper's licence.
- [ ] Funding: none (state "no external funding"). Competing interests: none.
- [ ] DECIDE: bioRxiv's FAQ asks authors to disclose use of generative AI tools in the manuscript. This conflicts
      with the standing rule that no file names an AI tool. Options: a generic disclosure sentence (no vendor named),
      or none if you judge it does not apply. Not added to the draft.
- [ ] No identifiable images of people (none in the paper).
- [ ] Order matters: bioRxiv does not post manuscripts already on another preprint server, so bioRxiv must come
      first if both are used. arXiv's endorsement page does not address prior posting elsewhere; check arXiv's
      general policy before cross-posting later.

## arXiv (cs.LG primary, q-bio.NC cross-list)

- [ ] Endorsement: a first submission to a category needs an endorser. Automatic endorsement requires an
      institutional email AND a claimed prior paper; with no prior arXiv papers, a personal endorser is needed.
      Route: start a submission to get the endorsement-request link, then ask one person you know (an NYU faculty
      member in data science or neural science). The page warns against mass-emailing endorsers.
- [ ] Add the NYU address to the arXiv account now (speeds endorsement).
- [ ] Source upload: `leakage_paper.tex` + `docs/figures/*.png` (arXiv prefers source over PDF).

## Kaggle

- [ ] Optional: a competition write-up post linking the preprint once posted (part of the Kaggle write-ups plan).
