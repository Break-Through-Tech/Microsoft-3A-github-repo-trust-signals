# External Repository Evaluation

Milestone 2, Task #4: score real repositories the models have never seen, compare the scores with public evidence, and find where the model is wrong.
The snapshot was taken on 2026-10-27 UTC.

A score describes how a repo's public signals compare with our 106 labeled examples.
It says nothing about intent, safety, or code quality, and a low score never means a repo is bad.
Only public metadata was read: no repository code was downloaded or run.

## Summary

- Of 20 repos, 5 are trusted, 2 uncertain, and 13 low confidence.
- The four large, active, long-established repos all score 92 or higher, as expected.
- **The score is mostly an age score.** It correlates with age at 0.78 (Spearman). None of the 12 repos under a year old is trusted: their median score is 13.7, against 85.5 for older repos. The cause is the labeled data, where 16 of the 17 repos under a year old are uncertain.
- **Newer and smaller repos are penalized, and at least some of that is unfair.** Four of the five labeled repos the model gets wrong, even when held out, are young repos labeled good: `openai/symphony`, `microsoft/markitdown`, `microsoft/graphrag`, and `anthropics/skills`.
- Three concrete modeling flaws explain most of the surprising scores: a spurious license rule, extrapolation on dormant repos, and an owner-type penalty on individuals.

## How the repos were chosen

Selection followed rules written before looking at any score, so it is not hand-picked.
Seven GitHub searches, each sorted by stars, found 277 unique repos, and 20 were selected across five groups.
Every repo found, with the reason it was excluded or selected, is in `data/external_candidates.csv`. The selected 20 are in `data/external_repos.csv`.

- **A, established and active:** created before 2018, pushed in the last two weeks, over 20,000 stars. Top 4.
- **B, established and inactive:** created before 2020, and either last pushed before 2024 (over 3,000 stars) or archived (over 10,000 stars). Top 2 from each search.
- **C, new, organization-owned:** created since January 2026, 1,000 to 10,000 stars. Four spaced evenly through the results.
- **D, new, viral, individually owned:** created in the last 90 days, over 3,000 stars. Top 4.
- **E, new, small, individually owned:** created since January 2026, 100 to 800 stars, more than 10 forks, pushed recently. Four spaced evenly through the results.

Forks were excluded from every search. Candidates were also dropped if they were already in the labeled dataset (13), had no code language or were Markdown, MDX, or TeX (33), were curated lists, tutorials, books, or translations (19), or failed a group's owner-type rule (55).
The 20 repos range from 9 to 6,268 days old and from 791 to 457,988 stars.
No repo overlaps the labeled dataset, which both the script and a test check.

## How they were scored

The models are the team's combined logistic regression and random forest from `notebooks/006_combined_model.ipynb`, on the same 15 signals, trained here on all 106 labeled repos.
The score is the mean of the two models' probability that a repo is good, from 0 to 100.
Bands are provisional until the trust score task defines official ones. I chose 70 and 40 from out-of-fold scores on the labeled repos alone, before scoring anything external:

- **Trusted (70 and up):** holds 71 of 82 good repos and 1 of 24 uncertain repos.
- **Uncertain (40 to 70):** holds 7 good and 5 uncertain repos. This is where the evidence is genuinely mixed.
- **Low confidence (under 40):** holds 4 of 82 good repos and 18 of 24 uncertain repos. It means little evidence of trust, not that a repo is bad.

A diagnostic score removes `age_days` and `stars_per_day` to show how much each score depends on age.
Reproduce everything with `uv run python -m gh_signals.external_eval`.

## Results

LR and RF are the two models' scores. Score is their mean.

| Group | Repo | Stars | Age (days) | LR | RF | Score | Band |
|---|---|---:|---:|---:|---:|---:|---|
| A | flutter/flutter | 180,383 | 4,252 | 100 | 99 | 99.5 | trusted |
| A | freeCodeCamp/freeCodeCamp | 457,988 | 4,324 | 100 | 85 | 92.4 | trusted |
| A | TheAlgorithms/Python | 225,553 | 3,754 | 100 | 84 | 92.0 | trusted |
| A | ohmyzsh/ohmyzsh | 190,179 | 6,268 | 100 | 84 | 92.0 | trusted |
| B | minio/minio | 61,300 | 4,303 | 87 | 71 | 79.1 | trusted |
| B | localstack/localstack | 65,135 | 3,653 | 68 | 28 | 48.3 | uncertain |
| B | NARKOZ/hacker-scripts | 49,858 | 3,992 | 0 | 61 | 30.4 | low_confidence |
| B | wg/wrk | 40,424 | 5,333 | 0 | 55 | 27.5 | low_confidence |
| C | unicity-sphere/sphere-sdk-connect-example | 9,503 | 254 | 89 | 2 | 45.5 | uncertain |
| C | cloudflare/moltworker | 9,946 | 272 | 24 | 18 | 21.0 | low_confidence |
| C | nexu-io/html-anything | 9,251 | 168 | 35 | 6 | 20.5 | low_confidence |
| C | cloudflare/computer | 9,722 | 143 | 28 | 1 | 14.4 | low_confidence |
| D | boykopovar/AnyPS5 | 48,670 | 84 | 0 | 79 | 39.5 | low_confidence |
| D | Ebony-Vinyl/dsh-our-free-model | 44,886 | 32 | 0 | 26 | 13.0 | low_confidence |
| D | Gathin23/cleanyourtv | 39,370 | 9 | 0 | 19 | 9.6 | low_confidence |
| D | NandhaKishorM/laya | 34,409 | 38 | 0 | 5 | 2.5 | low_confidence |
| E | coco-research/coco | 791 | 184 | 1 | 42 | 21.6 | low_confidence |
| E | Junchao-cs/SolarWM | 796 | 66 | 0 | 20 | 10.3 | low_confidence |
| E | OSideMedia/higgsfield-ai-prompt-skill | 800 | 232 | 1 | 10 | 5.9 | low_confidence |
| E | yuwen-cool/yuwen-publish-precheck | 794 | 103 | 1 | 8 | 4.4 | low_confidence |

Signal values behind those scores, in the same order. Typical labeled good repos have about 0.22 forks per star, 0.0197 watchers per star, and 0.029 issues per star (medians).
All 20 signals for each repo are in `data/external_features.csv`.

| Repo | Fork/star | Watchers/star | Issues/star | Stars/day | Days since push | Contributor activity | Ghost | Peak-day share | Burstiness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| flutter/flutter | 0.185 | 0.0196 | 0.0749 | 42.4 | 0 | 0.19 | 0.00 | 0.006 | 0.15 |
| freeCodeCamp/freeCodeCamp | 0.106 | 0.0187 | 0.0004 | 105.9 | 0 | 0.09 | 0.00 | 0.003 | 0.11 |
| TheAlgorithms/Python | 0.227 | 0.0263 | 0.0003 | 60.1 | 6 | 0.15 | 0.00 | 0.009 | 0.23 |
| ohmyzsh/ohmyzsh | 0.150 | 0.0141 | 0.0016 | 30.3 | 1 | 0.05 | 0.00 | 0.012 | 0.07 |
| minio/minio | 0.134 | 0.0106 | 0.0013 | 14.2 | 185 | 0.00 | 0.00 | 0.014 | 0.19 |
| localstack/localstack | 0.074 | 0.0002 | 0.0000 | 17.8 | 217 | 0.00 | 0.00 | 0.038 | 0.50 |
| NARKOZ/hacker-scripts | 0.133 | 0.0408 | 0.0014 | 12.5 | 1,099 | 0.00 | 0.10 | 0.074 | 0.78 |
| wg/wrk | 0.075 | 0.0180 | 0.0050 | 7.6 | 1,031 | 0.00 | 0.00 | 0.007 | 0.04 |
| unicity-sphere/sphere-sdk-connect-example | 0.003 | 0.0019 | 0.0000 | 37.4 | 5 | 0.67 | 0.00 | 0.450 | 0.78 |
| cloudflare/moltworker | 0.172 | 0.0046 | 0.0095 | 36.6 | 170 | 0.00 | 0.00 | 0.181 | 0.68 |
| nexu-io/html-anything | 0.095 | 0.0044 | 0.0064 | 55.1 | 13 | 0.32 | 0.00 | 0.149 | 0.43 |
| cloudflare/computer | 0.059 | 0.0029 | 0.0026 | 68.0 | 0 | 0.43 | 0.00 | 0.244 | 0.58 |
| boykopovar/AnyPS5 | 0.094 | 0.0085 | 0.0321 | 579.4 | 0 | 1.00 | 0.00 | 0.128 | 0.36 |
| Ebony-Vinyl/dsh-our-free-model | 0.028 | 0.0007 | 0.0079 | 1,402.7 | 0 | 1.00 | 0.00 | 0.076 | -0.10 |
| Gathin23/cleanyourtv | 0.120 | 0.0025 | 0.0060 | 4,374.4 | 0 | 1.00 | 0.00 | 0.173 | -0.37 |
| NandhaKishorM/laya | 0.090 | 0.0035 | 0.0032 | 905.5 | 0 | 1.00 | 0.00 | 0.187 | 0.25 |
| coco-research/coco | 0.018 | 0.0164 | 0.0885 | 4.3 | 0 | 0.88 | 0.00 | 0.142 | 0.50 |
| Junchao-cs/SolarWM | 0.052 | 0.0101 | 0.0038 | 12.1 | 3 | 1.00 | 0.00 | 0.255 | 0.42 |
| OSideMedia/higgsfield-ai-prompt-skill | 0.168 | 0.0037 | 0.0025 | 3.4 | 1 | 0.67 | 0.00 | 0.024 | -0.00 |
| yuwen-cool/yuwen-publish-precheck | 0.094 | 0.0025 | 0.0000 | 7.7 | 18 | 0.50 | 0.00 | 0.108 | 0.38 |

## Findings: where the model is wrong, and why

**1. The score is an age score, because the labeled data is.**
Among the 106 labeled repos, repos under a year old are 1 good and 16 uncertain, while repos over five years old are 70 good and 1 uncertain.
The model learned that young means uncertain. Out-of-fold, the four good repos that fall below 40 are `openai/symphony` (21.5), `microsoft/markitdown` (27.9), `anthropics/skills` (26.5), and `microsoft/graphrag` (37.6). All are under three years old and come from well-known organizations, so this is a false penalty with ground truth behind it.
Removing the age signals lifts `cloudflare/moltworker` only from 21.0 to 35.6 and `cloudflare/computer` from 14.4 to 18.9, so age is not the whole story for them.

**2. Dormant but healthy repos are scored by extrapolation.**
`wg/wrk` and `NARKOZ/hacker-scripts` have been quiet for 1,031 and 1,099 days. The longest-dormant good repo in training went 181 days, and the maximum for any labeled repo is 393.
The logistic regression puts them 17 and 18 standard deviations out, which drives its score to 0, while the random forest, which cannot extrapolate, gives 55 and 61.
Their engagement looks like a good repo's: 18.0 and 40.8 watchers per 1,000 stars, against a typical 19.7 for good repos. The low scores come from inactivity and individual ownership, not from any sign of manipulation, so they contradict the engagement evidence.

**3. The license rule is a one-sample artifact.**
Exactly one labeled repo has no license (`anthropics/skills`, which is good), so the logistic regression learned a negative license coefficient: having no license looks more trusted.
That is why `unicity-sphere/sphere-sdk-connect-example` scores 89 on logistic regression and 2 on the random forest. Flipping only its license flag moves the logistic regression score from 89.1 to 19.7.
It also gives `NARKOZ/hacker-scripts`, which has no license, a boost it does not deserve.

**4. Individually owned repos are penalized.**
In the labeled data, 11 of the 12 individually owned repos are uncertain, so the model treats individual ownership as a warning sign.
Removing the owner-type signal raises the four small individually owned repos (group E) by 8.4 to 13.3 points, and raises individually owned repos by 4.0 points on average.
This is the project's fairness concern in practice: individual maintainers are scored down for who they are, not for what they built.

**5. The models cannot see archived or abandoned status.**
`minio/minio` is archived yet scores trusted (79.1), while the also-archived `localstack/localstack` scores 48.3 and has only 0.2 watchers per 1,000 stars, the most unusual of its signals. A reader would want to know both are archived, and the score does not say so.

**6. Young repos from established companies look like viral repos.**
`cloudflare/computer` and `cloudflare/moltworker` are organization-owned and permissively licensed, and `cloudflare/computer` has 13 releases. But their watchers per 1,000 stars (2.9 and 4.6) and commit concentration (one person made 94% and 86% of the commits) match the uncertain pattern.
The current signals cannot tell a popular launch from an inflated one. An owner reputation signal is what would separate them.

**7. The two models disagree by 30 points or more on six repos.**
They are `boykopovar/AnyPS5` (0 vs 79), `NARKOZ/hacker-scripts` (0 vs 61), `wg/wrk` (0 vs 55), `coco-research/coco` (1 vs 42), `unicity-sphere/sphere-sdk-connect-example` (89 vs 2), and `localstack/localstack` (68 vs 28).
Disagreement is itself information. `boykopovar/AnyPS5` is the clearest case: it gained 48,670 stars in 84 days (579 per day) with 8.5 watchers per 1,000 stars, but also has 493 contributors, 4,590 forks, and 4 releases. The evidence conflicts, so it needs a human look, and its score of 39.5 sits just under the uncertain band.

**8. Group D matches the viral pattern, but that is not proof.**
These four repos gained 34,000 to 49,000 stars in 9 to 84 days, between 579 and 4,374 stars per day, with 0.7 to 8.5 watchers per 1,000 stars. That fits the uncertain examples, but several also show real activity (`NandhaKishorM/laya` has 48 releases and 99 contributors). The model can flag the pattern but cannot say whether it reflects inflation or genuine popularity.

## Are newer or smaller repos unfairly penalized?

Yes, in two ways. Young repos are penalized because the labeled data contains almost no young good repos, and small individually owned repos are penalized for owner type and low star counts.
All 12 repos under a year old score 45.5 or less, and all four small individual repos score between 4 and 22, even where their fork rates look healthy (`OSideMedia/higgsfield-ai-prompt-skill` has 0.17 forks per star against a typical 0.22).
We cannot say those repos are trustworthy, since there is no ground truth for them. The point is that the model cannot currently tell a good young repo from an uncertain one, and the one young good repo we can check with a held-out score, `openai/symphony`, is penalized heavily.

## Limitations

- There is no ground truth for the 20 repos, so findings compare scores with public metadata, not with verified labels. Twenty repos is a small sample.
- The labeled set is 106 repos with only 24 uncertain, and the uncertain ones skew young and individually owned.
- The 0 to 100 score and its bands are provisional, and the bands were chosen from the same labeled repos the models learn from.
- The models were trained on all 106 repos, so the earlier test-split AUCs do not describe this run.
- Star timing and the other time-dependent signals are tied to the snapshot date.

## Recommended improvements

1. **Label more repos, especially young good repos, dormant good repos, and individually owned good repos.** This is the largest fix, because most flaws above come from what the labeled set lacks.
2. **Add an owner reputation signal and show archived status.** This is what would separate Cloudflare's new repos from viral ones, and it addresses finding 5.
3. **Remove or tame brittle signals.** Drop `has_license`, which is near-constant, and cap or rank-transform `days_since_push` and `stars_per_day` so one extreme value cannot push a score to 0.
4. **Treat model disagreement and out-of-range inputs as uncertain,** and report the strongest reasons behind each score.
5. **Evaluate young repos separately,** so the age effect is measured instead of hidden in one overall AUC.
