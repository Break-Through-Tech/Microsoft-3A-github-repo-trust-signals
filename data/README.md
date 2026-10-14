# Dataset

`features.csv` is the one shared dataset: one row per labeled repo (106 repos, 82 good and 24 uncertain) with every signal from every feature branch.
`train_split.csv` and `test_split.csv` are its train and test rows (84 and 22, stratified 80/20, seed 42).
`manifest.json` records the snapshot time, split counts, column lists, missing values, and hashes of the label files and `features.csv`.
Everyone should evaluate against these files rather than building their own copies, so all results are comparable.

## Using it

Model inputs are the columns in `signal_columns` in the manifest, also available as `SIGNAL_COLUMNS` in `gh_signals.build_dataset`.
The target is `target_label` (1 is good, 0 is uncertain).
Never feed the identifier columns to a model: `full_name`, `listed_as`, `label`, `target_label`, and `split`.
Fit imputers and scalers on the train rows only.

`full_name` is the key across every file, as GitHub currently names the repo.
`listed_as` is the name as written in `good_repos.csv` and `uncertain.csv`, and differs for three repos that were renamed or re-cased.
Compare names case-insensitively.

## Signals

| Column | Meaning |
|---|---|
| `stargazers_count`, `forks_count`, `subscribers_count`, `open_issues_count`, `size` | Raw counts from the repo API (size is KB, open issues include pull requests) |
| `age_days`, `days_since_push` | Days from creation and from the last push to the snapshot time |
| `fork_star_ratio`, `subscriber_star_ratio`, `issues_per_star` | Forks, watchers, and open issues divided by stars |
| `stars_per_day` | Stars divided by age in days |
| `has_license`, `is_org_owned` | A license was detected; the owner is an organization |
| `contributor_activity_ratio` | Share of human contributors with a commit in the 12 weeks before the snapshot |
| `ghost_contributor_ratio` | Share of the top 10 human contributors by commits with zero followers and zero public repos |
| `star_to_follower_ratio` | Stars divided by the followers of the owner account |
| `peak_stars`, `max_day_share` | Stars on the busiest day, and that day's share of all stars |
| `active_day_ratio` | Share of days since creation with at least one star |
| `burstiness` | (std - mean) / (std + mean) of daily stars: near -1 is steady, near 1 is spiky |

Bot accounts are excluded from both contributor signals.
The star timing columns come from daily star history through the day before the snapshot.
There are no missing values in this snapshot, but a ratio is left empty (never 0) when it is undefined, so models should still impute.

## Rebuilding

```bash
uv run python -m gh_signals.build_dataset
```

Responses already in `data/raw/` are reused with no network access, and the output is byte-identical every time.
This takes about 5 minutes and 1.5 GB of memory, because the raw contributor data is large.
`data/raw/` is about 950 MB, so it is gitignored and not shared.
To take a fresh snapshot, delete `data/raw/` and rerun: this needs a GitHub token (`gh auth login` or `GITHUB_TOKEN`), makes about 2,800 requests, and resumes if interrupted.
A new snapshot changes the numbers, so commit all four output files together and tell the team to re-run their models.

Check that the committed files are consistent, with no network:

```bash
uv run python -m gh_signals.build_dataset --verify
```

`uv run pytest` runs the same check.
It fails if `good_repos.csv` or `uncertain.csv` change without a rebuild.

## Replaced files

`features.csv` replaces the earlier 20-repo sample of the same name, `star_features.csv`, `repo_features.csv`, and `raw_repos.json`.
Renamed columns: `repo_name` is now `full_name`, `fork_to_star_ratio` is now `fork_star_ratio`, and `issue_to_star_ratio` is now `issues_per_star`.
`notebooks/build_features.py` now just calls the builder.
`eval_results.csv` and `combined_model_results.csv` came from earlier snapshots and should be regenerated.
