"""A three-stage pipeline using chain mode with explicit intermediate
values.

Each stage returns a serializable basic container (a dict). The next
stage reads the previous stage's return value through the read-only
``pipeline_input`` argument. The value never touches ``ctx.obj`` and
never replaces a CLI parameter.

Example:

\b
    pipeline fetch transform report
"""

import click


@click.group(chain=True, pipeline=True)
def cli():
    """A three-stage data pipeline. Each stage feeds its return value
    into the next one."""


@cli.command()
def fetch():
    """Fetch raw records and return them as a mapping."""
    click.echo("fetch: collecting records")
    return {
        "records": [
            {"name": "alpha", "value": 3},
            {"name": "beta", "value": 7},
            {"name": "gamma", "value": 12},
        ]
    }


@cli.command()
@click.pipeline_input(dict)
def transform(pipeline_input):
    """Read the raw records and return a summary mapping."""
    records = pipeline_input["records"]
    click.echo(f"transform: processing {len(records)} records")
    return {
        "count": len(records),
        "total": sum(record["value"] for record in records),
        "names": [record["name"] for record in records],
    }


@cli.command()
@click.pipeline_input(dict)
def report(pipeline_input):
    """Read the summary and print a report."""
    click.echo("report:")
    click.echo(f"  count: {pipeline_input['count']}")
    click.echo(f"  total: {pipeline_input['total']}")
    click.echo(f"  names: {', '.join(pipeline_input['names'])}")


@cli.result_callback()
def process_results(results):
    """The result callback still receives the full list of stage
    returns, exactly like a regular chained group."""
    click.echo(f"pipeline finished with {len(results)} stages")
