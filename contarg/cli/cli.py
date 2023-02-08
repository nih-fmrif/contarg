import click
from .run_hierarchical import contarg

contarg = click.CommandCollection(sources=[contarg])

if __name__ == "__main__":
    contarg()
