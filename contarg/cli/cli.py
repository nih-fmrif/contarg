import click
from .run_hierarchical import contarg
from .run_seedmap import contarg as contargsm

contarg = click.CommandCollection(sources=[contarg, contargsm])

if __name__ == "__main__":
    contarg()
