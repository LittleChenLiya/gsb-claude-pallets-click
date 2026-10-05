"""A three-stage pipeline chain.

Each command returns a basic container value that the next command reads
through the read-only ``Context.pipeline_input`` attribute. ``ctx.obj``
and the CLI arguments of each command are left untouched.

Example:

.. code-block:: text

    $ pipeline load --path data.csv filter --column kind --keep x summarize
"""

import csv

import click


@click.group(chain=True, pipeline=True)
def cli():
    """Load records, transform them, and summarize the result."""


@cli.result_callback()
def report(stages):
    """The result callback still receives the list of all stage values."""
    click.echo(f"ran {len(stages)} stages")


@cli.command()
@click.option("--path", type=click.Path(exists=True), required=True)
def load(path):
    """Start the pipeline with the rows of a CSV file."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    click.echo(f"loaded {len(rows)} rows")
    return {"rows": rows}


@cli.command(pipeline_input_type=dict)
@click.option("--column", required=True)
@click.option("--keep", required=True)
@click.pass_context
def filter(ctx, column, keep):
    """Keep only the rows where COLUMN equals KEEP."""
    rows = [row for row in ctx.pipeline_input["rows"] if row[column] == keep]
    click.echo(f"kept {len(rows)} rows")
    return {"rows": rows}


@cli.command(pipeline_input_type=dict)
@click.pass_context
def summarize(ctx):
    """Print a summary of the incoming rows."""
    rows = ctx.pipeline_input["rows"]
    click.echo(f"summary: {len(rows)} rows, columns={sorted(rows[0]) if rows else []}")
    return {"count": len(rows)}


if __name__ == "__main__":
    cli()
