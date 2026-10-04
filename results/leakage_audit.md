# Leakage audit

Split: `src/split.py` (EVAL_SUBJECTS=(3, 6, 9, 12), GUARD=1.0 s)

- PASS: subject axis (val=True runs): 0 training windows from selection subjects (1, 5, 10)
- PASS: selection and eval subjects are disjoint
- PASS: subject axis: 0 training windows from eval subjects (3, 6, 9, 12)
- PASS: stimulus axis (shared Sherlock1 ses-11/12): min gap between any training window end and block B start = 1.43 s (>= 1 s required)
- INFO: training windows 348824; block-B eval windows 2532 over 4 subjects
- PASS: text axis: 0 of 1643 block-B word 8-grams also occur in training text
- PASS: holdout metadata: no reference to word_onsets_s / shuffle_seed / onset_s in src/ or scripts/
