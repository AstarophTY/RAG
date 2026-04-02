import fire
from sys import stderr
import sys

from src.__main__ import Pipeline, _validate_no_extra_flags


if __name__ == "__main__":
    try:
        _validate_no_extra_flags(sys.argv[1:])
        fire.Fire(Pipeline)
    except Exception as e:
        print(e, file=stderr)
