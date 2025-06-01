import git


def get_repo_dir():
    return git.Repo(".", search_parent_directories=True).working_tree_dir
