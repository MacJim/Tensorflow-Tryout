"""
Changes in `os.environ` are effective only for the current process and are reset after that.

Doesn't seem to work with `LD_LIBRARY_PATH`.
"""

import os


PATHS_TO_ADD = [
    # ("/usr/local/cuda-10.2/bin", "PATH"),
    ("/scratch/MJ/cuda/cudnn-10.0-v7.6.5.32/lib64", "LD_LIBRARY_PATH"),
    # ("/scratch/MJ/cuda/cudnn-10.0-v7.6.5.32/include", "INCLUDE"),
]


# MARK: - Remove/add paths
def remove_path_from_paths(path: str, paths: str, separator=":") -> str:
    pass


def add_path_to_paths(path: str, paths: str, separator=":") -> str:
    return f"{path}:{paths}"


def remove_trailing_separator_from_paths(paths: str, separator=":") -> str:
    while (paths.endswith(separator)):
        paths = paths[:-1]
    
    return paths


# MARK: - Main
def main():
    for path, env_name in PATHS_TO_ADD:
        # print(os.environ[env_name])
        os.environ[env_name] = add_path_to_paths(path, os.environ[env_name])
        os.environ[env_name] = remove_trailing_separator_from_paths(os.environ[env_name])
        # print(os.environ[env_name])


if (__name__ == "__main__"):
    # MARK: Switch to current dir
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"Working directory: {os.getcwd()}")

    # path = os.environ["PATH"]
    # print(type(path), path)    # Just a comma-separated string.
    # os.environ["MJ_TEST"] = "TEST_STR"
    # print("PATH:", os.environ["PATH"])
    print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])
    # print("INCLUDE:", os.environ["INCLUDE"])

    main()

    # print("PATH:", os.environ["PATH"])
    print("LD_LIBRARY_PATH:", os.environ["LD_LIBRARY_PATH"])
    # print("INCLUDE:", os.environ["INCLUDE"])
