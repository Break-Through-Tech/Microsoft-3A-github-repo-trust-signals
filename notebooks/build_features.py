import sys
import os
import json
import glob
import pandas as pd

# search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from gh_signals.signals import (
    issues_to_stars_ratio,
    stars_to_followers_ratio,
    fork_to_stars_ratio,
)

def process_repository_data(raw_repo_data: dict) -> dict:
    """
    Takes GitHub API JSON metadata and computes target features.
    """

    name = raw_repo_data.get("full_name")

    forks = raw_repo_data.get("forks_count", 0)
    stargazers = raw_repo_data.get("stargazers_count", 0)
    open_issues = raw_repo_data.get("open_issues_count", 0)

    owner_info = raw_repo_data.get("owner", {})
    owner_followers = owner_info.get("followers", 0)

    return {
        "repo_name": name,
        "forks_count": forks,
        "stargazers_count": stargazers,
        "open_issues_count": open_issues,
        "issue_to_star_ratio": issues_to_stars_ratio(open_issues, stargazers),
        "star_to_follower_ratio": stars_to_followers_ratio(stargazers, owner_followers),
        "fork_to_star_ratio": fork_to_stars_ratio(forks, stargazers)
    }

def load_raw_data() -> list[dict]:
    """Loads JSON repo metadata"""
    if os.path.exists("data/raw_repos.json"):
        with open("data/raw_repos.json", "r") as f:
            return json.load(f)
            
    raw_repos = []
    for file_path in glob.glob("data/raw/*.json"):
        with open(file_path, "r") as f:
            raw_repos.append(json.load(f))
            
    return raw_repos

if __name__ == "__main__":
    raw_repos_json = load_raw_data()
    df = pd.DataFrame([process_repository_data(repo) for repo in raw_repos_json])
    df.to_csv("data/features.csv", index=False)