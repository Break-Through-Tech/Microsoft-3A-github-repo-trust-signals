"""
Signal functions for GitHub Repo Trust Signals.
Each function takes a repo and returns a numeric value.
"""

def stars_to_followers_ratio(repo_data: dict) -> float | None:
    """
    Ratio of repo stars to the owner's follower count.
    High stars with low followers can indicate artificial stars.
    Returns None if data is missing.
    """
    stars = repo_data.get("stargazers_count")
    followers = repo_data.get("owner", {}).get("followers")
    
    if stars is None or followers is None or followers == 0:
        return None
    return stars / followers


def issues_per_star(repo_data: dict) -> float | None:
    """
    Ratio of open issues to stars.
    Low values may indicate low engagement.
    Returns None if data is missing.
    """
    issues = repo_data.get("open_issues_count")
    stars = repo_data.get("stargazers_count")
    
    if issues is None or stars is None or stars == 0:
        return None
    return issues / stars