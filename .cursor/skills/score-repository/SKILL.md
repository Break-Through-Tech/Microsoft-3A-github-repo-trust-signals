---
name: score-repository
description: >-
  Score a public GitHub repository from metadata and explain the 0 to 100 trust
  score, tier, and signal breakdown. Use when the user asks for a repository
  trust score, whether a GitHub repo looks established, or to run trust-score.
---

# Score a repository

Read public GitHub metadata only. Do not clone, download, or run the repository.

## Library

```python
import asyncio
from gh_signals.score import score_repository

result = asyncio.run(score_repository("https://github.com/owner/name"))
```

`result.score` is an integer from 0 to 100, or `None` when the number is withheld. `result.tier` is `established`, `mixed`, or `limited_evidence`. `result.signals` is the breakdown. `result.usage_summary` says how long the score took, how many GitHub requests it used, and how many remain this hour.

A repeat of the same repository within 24 hours is served from cache and does not spend the quota. Without a token the limit is 60 requests an hour. With `GITHUB_TOKEN` or `gh auth login` it is 5,000. A new repository uses one request for the repo, one for the owner, and one per page of star history. Star history stops after 4 seconds.

## CLI

From this repository:

```bash
uv run trust-score https://github.com/owner/name
uv run trust-score owner/name --annotate too-low --comment "why the score looks off"
```

`--annotate` accepts `about-right`, `too-high`, or `too-low`. The note is appended to `data/feedback.jsonl`. Training does not read that file.

## How to read a result

- Established is 70 to 100. Mixed is 40 to 69. Limited evidence is under 40, or the number is withheld.
- The number is withheld under 90 days old or under 50 stars.
- A star count of at least 1,000 does not lower the score. The training set averages about 52,500 stars, so a popular repository can sit below that center.
- Stars per day, the busiest day, and burstiness can still lower a score. Those are timing signals, not the raw star count.
- Do not describe a low score as harmful or malicious. It means limited public evidence.
