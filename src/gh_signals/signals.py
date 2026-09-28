"""
Signal functions for GitHub Repo Trust Signals.
Each function takes a repo and returns a numeric value.
"""

def stars_to_followers_ratio(stars: int, followers: int) -> float | None:
    """
    Ratio of repo stars to follower count.
    High stars with low followers can indicate artificial stars.
    Returns None if data is missing or followers == 0.
    """
    
    if stars is None or followers is None or followers == 0:
        return None
        
    return float(stars / followers)


def issues_to_stars_ratio(issues: int, stars: int) -> float | None:
    """
    Ratio of open issues to stars.
    Low values may indicate low engagement.
    Returns None if data is missing or stars == 0.
    """
    
    if issues is None or stars is None or stars == 0:
        return None
        
    return float(issues / stars)

def fork_to_stars_ratio(forks: int, stars: int) -> float | None:
    """
    Ratio of forks to repository stars.

    Returns None if forks or star data is missing or 0,
    because division by 0 is undefined
    """

    if forks is None or stars is None or stars == 0:
        return None

    return float (forks / stars)
