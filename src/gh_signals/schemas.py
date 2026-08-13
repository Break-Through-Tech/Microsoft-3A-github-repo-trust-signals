"""Typed models for the parts of the GitHub API we consume."""

from pydantic import BaseModel


class Repo(BaseModel):
    """The repository fields we use as trust signals, named exactly as the GitHub API names them."""

    full_name: str
    stargazers_count: int
    forks_count: int
