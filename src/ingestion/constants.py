CODE_EXTENSIONS: set[str] = {
    ".py",
    ".cu",
    ".cuh",
    ".cpp",
    ".hpp",
    ".h",
    ".cc",
    ".c",
}

DOCS_EXTENSIONS: set[str] = {
    ".md",
    ".rst",
    ".txt",
}

ALL_EXTENSIONS: set[str] = CODE_EXTENSIONS | DOCS_EXTENSIONS

IGNORED_DIRS: set[str] = {
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".eggs",
    "*.egg-info",
    ".tox",
    "htmlcov",
    ".hypothesis",
}

IGNORED_FILENAMES: set[str] = {
    "conftest.py",
    "setup.py",
    "setup.cfg",
}
