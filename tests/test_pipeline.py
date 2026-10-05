import pytest

import click


def test_pipeline_mapping_passed_to_next_command(runner):
    """A mapping returned by the first stage is readable through the
    read-only ``pipeline_input`` of the second stage.
    """
    seen = []

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def produce():
        return {"key": "value"}

    @cli.command(pipeline_input_type=dict)
    @click.pass_context
    def consume(ctx):
        seen.append(ctx.pipeline_input)
        click.echo(ctx.pipeline_input["key"])

    result = runner.invoke(cli, ["produce", "consume"])
    assert result.exception is None
    assert seen == [{"key": "value"}]
    assert result.output.splitlines() == ["value"]


def test_pipeline_first_stage_gets_empty_input(runner):
    """The first command of each invocation starts from an empty input."""
    seen = []

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    @click.pass_context
    def first(ctx):
        seen.append(ctx.pipeline_input)
        return 1

    @cli.command()
    @click.pass_context
    def second(ctx):
        seen.append(ctx.pipeline_input)

    result = runner.invoke(cli, ["first", "second"])
    assert result.exception is None
    assert seen == [None, 1]


def test_pipeline_wrong_type_skips_consumer_callback(runner):
    """If the previous stage produces the wrong type, the consumer
    contract fails before the business callback runs, and ``ctx.obj``
    is left untouched.
    """
    calls = []
    obj = {"marker": True}
    seen_obj = []

    @click.group(chain=True, pipeline=True)
    @click.pass_context
    def cli(ctx):
        ctx.obj = obj

    @cli.command()
    @click.pass_context
    def produce(ctx):
        seen_obj.append(ctx.obj)
        return ["not", "a", "mapping"]

    @cli.command(pipeline_input_type=dict)
    def consume():
        calls.append("consume")

    result = runner.invoke(cli, ["produce", "consume"])
    assert result.exit_code != 0
    assert "expected pipeline input of type dict" in result.output
    # The consumer callback never ran.
    assert calls == []
    # The pipeline never touched ctx.obj.
    assert seen_obj == [obj]
    assert obj == {"marker": True}


def test_pipeline_close_callbacks_called_once(runner):
    """Every stage still closes its own context resources exactly once."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    closed = []

    @cli.command()
    @click.pass_context
    def first(ctx):
        ctx.call_on_close(lambda: closed.append("first"))
        return {"a": 1}

    @cli.command(pipeline_input_type=dict)
    @click.pass_context
    def second(ctx):
        ctx.call_on_close(lambda: closed.append("second"))
        return {"b": 2}

    result = runner.invoke(cli, ["first", "second"])
    assert result.exception is None
    assert closed == ["first", "second"]


def test_pipeline_failure_closes_entered_contexts_once(runner):
    """On contract failure, stages that already ran are closed exactly
    once and the failed stage's callback never runs.
    """
    closed = []
    calls = []

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    @click.pass_context
    def first(ctx):
        ctx.call_on_close(lambda: closed.append("first"))
        return "not-a-mapping"

    @cli.command(pipeline_input_type=dict)
    def second():
        calls.append("second")

    result = runner.invoke(cli, ["first", "second"])
    assert result.exit_code != 0
    assert calls == []
    assert closed == ["first"]


def test_pipeline_reinvocation_starts_from_empty_input(runner):
    """Reusing the group after a successful or failed run leaves no
    residual intermediate values behind.
    """
    seen = []

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    @click.pass_context
    def first(ctx):
        seen.append(ctx.pipeline_input)
        return {"n": 1}

    @cli.command(pipeline_input_type=dict)
    @click.pass_context
    def second(ctx):
        seen.append(ctx.pipeline_input)

    @cli.command("fail")
    @click.pass_context
    def fail_cmd(ctx):
        seen.append(ctx.pipeline_input)
        raise click.UsageError("boom")

    for _ in range(2):
        result = runner.invoke(cli, ["first", "second"])
        assert result.exception is None

    # A failed run does not leak state into the next one either.
    result = runner.invoke(cli, ["first", "fail"])
    assert result.exit_code != 0
    result = runner.invoke(cli, ["first", "second"])
    assert result.exception is None

    assert seen == [
        None,
        {"n": 1},
        None,
        {"n": 1},
        None,
        {"n": 1},
        None,
        {"n": 1},
    ]


def test_pipeline_result_callback_shape_preserved(runner):
    """The result callback still receives the list of all return values."""
    seen = []

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def a():
        return {"a": 1}

    @cli.command(pipeline_input_type=dict)
    @click.pass_context
    def b(ctx):
        return {"b": ctx.pipeline_input["a"] + 1}

    @cli.result_callback()
    def process(values):
        seen.append(values)

    result = runner.invoke(cli, ["a", "b"])
    assert result.exception is None
    assert seen == [[{"a": 1}, {"b": 2}]]


def test_pipeline_does_not_replace_cli_args(runner):
    """Pipeline input is separate from the command's CLI arguments."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def produce():
        return {"x": 1}

    @cli.command(pipeline_input_type=dict)
    @click.argument("name")
    @click.pass_context
    def greet(ctx, name):
        click.echo(f"{name}:{ctx.pipeline_input['x']}")

    result = runner.invoke(cli, ["produce", "greet", "world"])
    assert result.exception is None
    assert result.output.splitlines() == ["world:1"]


def test_pipeline_input_is_read_only(runner):
    """Assigning to ``pipeline_input`` is not allowed."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    @click.pass_context
    def first(ctx):
        with pytest.raises(AttributeError):
            ctx.pipeline_input = {}
        return {}

    @cli.command(pipeline_input_type=dict)
    def second():
        pass

    result = runner.invoke(cli, ["first", "second"])
    assert result.exception is None


def test_pipeline_rejects_non_container_values(runner):
    """Only picklable basic container values may flow through a pipeline."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def produce():
        return object()

    @cli.command()
    def consume():
        pass

    result = runner.invoke(cli, ["produce", "consume"])
    assert result.exit_code != 0
    assert "cannot be passed through a pipeline" in result.output


def test_pipeline_requires_chain():
    with pytest.raises(RuntimeError, match="pipeline mode"):
        click.Group("cli", pipeline=True)


def test_pipeline_three_stage_example(runner):
    """Three-stage example: load -> increment -> increment -> show."""

    @click.group(chain=True, pipeline=True)
    def cli():
        """Load a record, transform it, and show the result."""

    @cli.command()
    def load():
        """Start the pipeline with a record."""
        return {"count": 1}

    @cli.command(pipeline_input_type=dict)
    @click.pass_context
    def increment(ctx):
        """Increment the count of the incoming record."""
        record = dict(ctx.pipeline_input)
        record["count"] += 1
        return record

    @cli.command(pipeline_input_type=dict)
    @click.pass_context
    def show(ctx):
        """Print the incoming record."""
        click.echo(f"count={ctx.pipeline_input['count']}")

    result = runner.invoke(cli, ["load", "increment", "increment", "show"])
    assert result.exception is None
    assert result.output.splitlines() == ["count=3"]
