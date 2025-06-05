from pathlib import Path

import git


def get_repo_dir() -> Path:
    return Path(git.Repo(".", search_parent_directories=True).working_tree_dir)
