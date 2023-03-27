import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-big",
        action="store_true",
        default=False,
        help="Run big tests which fail on github",
    )
