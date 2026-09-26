import sys
import os
import pandas as pd

# search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gh_signals.signals import (
    issues_per_star,
    stars_to_followers_ratio,
)

def process_repository_data(raw_repo_data: dict) -> dict:
    """
    Takes GitHub API JSON metadata and computes target features.
    """
    stargazers = raw_repo_data.get("stargazers_count", 0)
    open_issues = raw_repo_data.get("open_issues_count", 0)
    
    owner_info = raw_repo_data.get("owner", {})
    owner_followers = owner_info.get("followers", 0)

    return {
        "repo_name": raw_repo_data.get("full_name"),
        "stargazers_count": stargazers,
        "open_issues_count": open_issues,
        "issues_per_star": issues_per_star(open_issues, stargazers),
        "stars_to_followers_ratio": stars_to_followers_ratio(stargazers, owner_followers),
    }

# Exporting to features.csv, raw_repos_json needs to be loaded from the JSON files or API output from Task 1 so commented out for now
# df = pd.DataFrame([process_repository_data(repo) for repo in raw_repos_json])
# df.to_csv("data/features.csv", index=False)