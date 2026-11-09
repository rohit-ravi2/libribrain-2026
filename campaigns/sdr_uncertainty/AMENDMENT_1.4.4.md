# Amendment 1.4.4 — honesty merge

Protocol id of the parent: `strong-decoder-noise-1.4`. This file is amendment `1.4.4`. `CAMPAIGN.md` and amendments 1.4, 1.4.2, and 1.4.3 are left as they were. The noise equation, the estimand, the candidate list, the in-vocabulary filter, the day-term rule, the memory cap, and the scheduler lock are unchanged. There is no runtime ceiling. `effect_claimed` stays false. `valid_scientific_claim` stays false.

## What failed

`noise-fit` had already been written with `day_terms_inestimable: true`. Eight forward chunks had succeeded. The next launch was `score-frozen`. It raised before any score file was written:

```text
TypeError: honesty() got multiple values for keyword argument 'not_confirmatory'
```

`score_frozen_directory` returns `not_confirmatory: true` and `independent_confirmation_required: true`. `measurement_status` returns the same two keys, also true. `run_next` called `honesty(**payload, **status, checkpoint=ckpt)`. Python binds the keywords before `honesty` runs, so the duplicate name is a TypeError even though the values agree and even though `honesty` would have pinned both flags afterwards. This stage had not been run end to end, so the same class of call was still present on `score-partial` (`honesty(**one_dict)` is safe) and on `smoke` (`honesty(**info, stage=..., next_action=...)`, which collides if `info` ever carries `stage` or `next_action`).

`report.update(info)` on `partial-train` is not a TypeError. It is the same merge class: a later mapping can replace a flag that was already false.

## Rule

Do not splat two mappings into `honesty`. Pass them as separate sources. `merge_honesty_sources` builds one mapping:

- one source, or several sources with equal values: keep that value
- a claim flag whose sources disagree: keep the value that does not advance a claim
- any other disagreement, including a metric or an integrity flag such as `dates_invented`, `competition_holdout_read`, or `temperature_applied_to_reported_logits`: raise `ProtocolError`

Claim flags and the value that does not advance a claim:

| flag | kept on disagreement |
| --- | --- |
| `effect_claimed` | false |
| `valid_scientific_claim` | false |
| `real_meg_scientific_validation` | false |
| `accuracy_claimed_from_temperature` | false |
| `frozen_trunk_subject0_days_are_holdout` | false |
| `eval_subjects_are_pristine_confirmatory_holdout` | false |
| `lpp_pooled` | false |
| `pooled_with_libribrain` | false |
| `full_retrain_launched` | false |
| `exploratory` | true |
| `not_confirmatory` | true |
| `independent_confirmation_required` | true |

`honesty` still pins those of the above that it already pinned, plus `dates_invented`, `session_ordinal_used_as_day`, and `competition_holdout_read` to false. `measurement_complete` is not a claim. Sources that disagree on it raise rather than turning a fixture into a finished measurement or hiding one. Agreement on a claim-ward value is not rewritten by the merge; the pin inside `honesty` still clears the flags `honesty` is responsible for.

`score-frozen`, `score-partial`, `partial-train`, the other `run_next` report updates, and `smoke` all go through this merge. Forward caches are not rewritten. `protocol.py` and its hash are unchanged.
