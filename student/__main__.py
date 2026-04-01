import fire
from sys import stderr

from src.__main__ import Pipeline


if __name__ == "__main__":
    try:
        fire.Fire(Pipeline)
    except Exception as e:
        print(e, file=stderr)
