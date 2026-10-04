# DRAFT — NOT SENT. Rohit decides whether and how to send (libribrain@robots.ox.ac.uk or the Discord).

**To:** libribrain@robots.ox.ac.uk
**Subject:** PNPL 2026 Broad track: clarification on unlabelled test-time adaptation

Dear LibriBrain organisers,

Thank you for running the 2026 competition, and for the Sep 18 note on cross-track leakage. We have one clarification question about the Broad track before we finalise our pipeline.

Is it permitted to adapt a model to a subject using **only that subject's own unlabelled holdout MEG**? Examples would be fitting a per-subject sensor alignment, or normalisation statistics, from the subject's holdout windows. This would be done:
- without any labels;
- without `word_onsets_s` absolute timing or `shuffle_seed`;
- without any correspondence between rows of different subjects (each subject is adapted from its own windows alone).

This matters most for the zero-shot subjects (33–39), whose only available data is the holdout. The official tutorial already standardises holdout windows. We would like to know whether going beyond per-channel statistics, for example with a learned linear sensor map fitted unsupervised, is within the rules.

We will keep a fallback submission that uses no test-time adaptation in any case.

Best regards,
Rohit Ravi
