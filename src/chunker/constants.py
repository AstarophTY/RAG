CODE_EXTENSIONS: set[str] = {
    ".py",
}

DOCS_EXTENSIONS: set[str] = {
    ".md",
    ".rst",
    ".txt",
}

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

IGNORED_FILES: set[str] = {
    "setup.py"
}

ALL_EXTENSIONS: set[str] = CODE_EXTENSIONS | DOCS_EXTENSIONS
