import sys

import pytest

import click


def debug():
    click.echo(
        f"{sys._getframe(1).f_code.co_name}"
        f"={'|'.join(click.get_current_context().args)}"
    )


def test_basic_chaining(runner):
    @click.group(chain=True)
    def cli():
        pass

    @cli.command("sdist")
    def sdist():
        click.echo("sdist called")

    @cli.command("bdist")
    def bdist():
        click.echo("bdist called")

    result = runner.invoke(cli, ["bdist", "sdist", "bdist"])
    assert not result.exception
    assert result.output.splitlines() == [
        "bdist called",
        "sdist called",
        "bdist called",
    ]


@pytest.mark.parametrize(
    ("args", "expect"),
    [
        (["--help"], "COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]..."),
        (["--help"], "ROOT HELP"),
        (["sdist", "--help"], "SDIST HELP"),
        (["bdist", "--help"], "BDIST HELP"),
        (["bdist", "sdist", "--help"], "SDIST HELP"),
    ],
)
def test_chaining_help(runner, args, expect):
    @click.group(chain=True)
    def cli():
        """ROOT HELP"""
        pass

    @cli.command("sdist")
    def sdist():
        """SDIST HELP"""
        click.echo("sdist called")

    @cli.command("bdist")
    def bdist():
        """BDIST HELP"""
        click.echo("bdist called")

    result = runner.invoke(cli, args)
    assert not result.exception
    assert expect in result.output


def test_chaining_with_options(runner):
    @click.group(chain=True)
    def cli():
        pass

    @cli.command("sdist")
    @click.option("--format")
    def sdist(format):
        click.echo(f"sdist called {format}")

    @cli.command("bdist")
    @click.option("--format")
    def bdist(format):
        click.echo(f"bdist called {format}")

    result = runner.invoke(cli, ["bdist", "--format=1", "sdist", "--format=2"])
    assert not result.exception
    assert result.output.splitlines() == ["bdist called 1", "sdist called 2"]


@pytest.mark.parametrize(("chain", "expect"), [(False, "1"), (True, "[]")])
def test_no_command_result_callback(runner, chain, expect):
    """When a group has ``invoke_without_command=True``, the result
    callback is always invoked. A regular group invokes it with
    its return value, a chained group with ``[]``.
    """

    @click.group(invoke_without_command=True, chain=chain)
    def cli():
        return 1

    @cli.result_callback()
    def process_result(result):
        click.echo(result, nl=False)

    result = runner.invoke(cli, [])
    assert result.output == expect


def test_chaining_with_arguments(runner):
    @click.group(chain=True)
    def cli():
        pass

    @cli.command("sdist")
    @click.argument("format")
    def sdist(format):
        click.echo(f"sdist called {format}")

    @cli.command("bdist")
    @click.argument("format")
    def bdist(format):
        click.echo(f"bdist called {format}")

    result = runner.invoke(cli, ["bdist", "1", "sdist", "2"])
    assert not result.exception
    assert result.output.splitlines() == ["bdist called 1", "sdist called 2"]


@pytest.mark.parametrize(
    ("args", "input", "expect"),
    [
        (["-f", "-"], "foo\nbar", ["foo", "bar"]),
        (["-f", "-", "strip"], "foo \n bar", ["foo", "bar"]),
        (["-f", "-", "strip", "uppercase"], "foo \n bar", ["FOO", "BAR"]),
    ],
)
def test_pipeline(runner, args, input, expect):
    @click.group(chain=True, invoke_without_command=True)
    @click.option("-f", type=click.File("r"))
    def cli(f):
        pass

    @cli.result_callback()
    def process_pipeline(processors, f):
        iterator = (x.rstrip("\r\n") for x in f)
        for processor in processors:
            iterator = processor(iterator)
        for item in iterator:
            click.echo(item)

    @cli.command("uppercase")
    def make_uppercase():
        def processor(iterator):
            for line in iterator:
                yield line.upper()

        return processor

    @cli.command("strip")
    def make_strip():
        def processor(iterator):
            for line in iterator:
                yield line.strip()

        return processor

    result = runner.invoke(cli, args, input=input)
    assert not result.exception
    assert result.output.splitlines() == expect


def test_args_and_chain(runner):
    @click.group(chain=True)
    def cli():
        debug()

    @cli.command()
    def a():
        debug()

    @cli.command()
    def b():
        debug()

    @cli.command()
    def c():
        debug()

    result = runner.invoke(cli, ["a", "b", "c"])
    assert not result.exception
    assert result.output.splitlines() == ["cli=", "a=", "b=", "c="]


def test_group_arg_behavior(runner):
    with pytest.raises(RuntimeError):

        @click.group(chain=True)
        @click.argument("forbidden", required=False)
        def bad_cli():
            pass

    with pytest.raises(RuntimeError):

        @click.group(chain=True)
        @click.argument("forbidden", nargs=-1)
        def bad_cli2():
            pass

    @click.group(chain=True)
    @click.argument("arg")
    def cli(arg):
        click.echo(f"cli:{arg}")

    @cli.command()
    def a():
        click.echo("a")

    result = runner.invoke(cli, ["foo", "a"])
    assert not result.exception
    assert result.output.splitlines() == ["cli:foo", "a"]


@pytest.mark.xfail
def test_group_chaining(runner):
    @click.group(chain=True)
    def cli():
        debug()

    @cli.group()
    def l1a():
        debug()

    @l1a.command()
    def l2a():
        debug()

    @l1a.command()
    def l2b():
        debug()

    @cli.command()
    def l1b():
        debug()

    result = runner.invoke(cli, ["l1a", "l2a", "l1b"])
    assert not result.exception
    assert result.output.splitlines() == ["cli=", "l1a=", "l2a=", "l1b="]


# ---------------------------------------------------------------------------
# Pipeline mode
# ---------------------------------------------------------------------------


def test_pipeline_basic_mapping(runner):
    """A first stage returning a mapping is read by the next stage."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"name": "click", "value": 42}

    @cli.command()
    @click.pipeline_input(dict)
    def stage2(pipeline_input):
        click.echo(f"{pipeline_input['name']}={pipeline_input['value']}")

    result = runner.invoke(cli, ["stage1", "stage2"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["click=42"]


def test_pipeline_result_callback_shape_preserved(runner):
    """The result callback still receives the full list of stage returns."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"a": 1}

    @cli.command()
    @click.pipeline_input(dict)
    def stage2(pipeline_input):
        return {"b": 2}

    @cli.result_callback()
    def process(results):
        click.echo(type(results).__name__)
        click.echo(results)

    result = runner.invoke(cli, ["stage1", "stage2"])
    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "list",
        "[{'a': 1}, {'b': 2}]",
    ]


def test_pipeline_wrong_type_callback_not_called(runner):
    """A type mismatch is raised before the consumer callback runs."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return [1, 2, 3]

    calls = []

    @cli.command()
    @click.pipeline_input(dict)
    def stage2(pipeline_input):
        calls.append("stage2")

    result = runner.invoke(cli, ["stage1", "stage2"])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)
    assert calls == []


def test_pipeline_wrong_type_ctx_obj_unchanged(runner):
    """The pipeline value never replaces or pollutes ``ctx.obj``."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"secret": "pipeline"}

    @cli.command()
    @click.pipeline_input(dict)
    @click.pass_obj
    def stage2(obj, pipeline_input):
        click.echo(f"obj={obj}")
        click.echo(f"pipeline={pipeline_input}")

    result = runner.invoke(cli, ["stage1", "stage2"], obj={"user": "obj"})
    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "obj={'user': 'obj'}",
        "pipeline={'secret': 'pipeline'}",
    ]


def test_pipeline_close_callbacks_called_once(runner):
    """Each stage's own context resources are closed on exit."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    closed = []

    @cli.command()
    def stage1():
        @click.pass_context
        def register(ctx):
            ctx.call_on_close(lambda: closed.append("stage1"))

        register()
        return {"a": 1}

    @cli.command()
    def stage2():
        @click.pass_context
        def register(ctx):
            ctx.call_on_close(lambda: closed.append("stage2"))

        register()
        return {"b": 2}

    result = runner.invoke(cli, ["stage1", "stage2"])
    assert result.exit_code == 0
    assert closed == ["stage1", "stage2"]


def test_pipeline_close_called_on_failure(runner):
    """A later failure still closes earlier stages' resources."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    closed = []

    @cli.command()
    def stage1():
        @click.pass_context
        def register(ctx):
            ctx.call_on_close(lambda: closed.append("stage1"))

        register()
        return {"a": 1}

    @cli.command()
    @click.pipeline_input(dict)
    def stage2(pipeline_input):
        raise RuntimeError("boom")

    result = runner.invoke(cli, ["stage1", "stage2"])
    assert result.exit_code == 1
    assert closed == ["stage1"]


def test_pipeline_reuse_starts_from_empty_input(runner):
    """Reusing the group after a run starts from empty pipeline input."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"n": 1}

    @cli.command()
    @click.pipeline_input(dict)
    def stage2(pipeline_input):
        click.echo(pipeline_input)

    # A successful run followed by another successful run.
    first = runner.invoke(cli, ["stage1", "stage2"])
    assert first.exit_code == 0
    second = runner.invoke(cli, ["stage1", "stage2"])
    assert second.exit_code == 0

    # A consumer as the first stage has no input, so it fails. This proves
    # no intermediate value leaked from the previous run.
    third = runner.invoke(cli, ["stage2"])
    assert third.exit_code == 1
    assert isinstance(third.exception, TypeError)


def test_pipeline_three_stages(runner):
    """The intermediate value flows through three stages."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"a": 1}

    @cli.command()
    @click.pipeline_input(dict)
    def stage2(pipeline_input):
        click.echo(f"stage2 got {pipeline_input}")
        return {"b": 2}

    @cli.command()
    @click.pipeline_input(dict)
    def stage3(pipeline_input):
        click.echo(f"stage3 got {pipeline_input}")

    result = runner.invoke(cli, ["stage1", "stage2", "stage3"])
    assert result.exit_code == 0
    assert result.output.splitlines() == [
        "stage2 got {'a': 1}",
        "stage3 got {'b': 2}",
    ]


def test_pipeline_stage_without_decorator_still_flows(runner):
    """A stage that does not consume still passes its return on."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"a": 1}

    @cli.command()
    def stage2():  # no pipeline_input decorator
        return {"b": 2}

    @cli.command()
    @click.pipeline_input(dict)
    def stage3(pipeline_input):
        click.echo(pipeline_input)

    result = runner.invoke(cli, ["stage1", "stage2", "stage3"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["{'b': 2}"]


def test_pipeline_non_serializable_rejected(runner):
    """Only serializable basic container values may flow through."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return lambda: None

    result = runner.invoke(cli, ["stage1"])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)


def test_pipeline_nested_non_serializable_rejected(runner):
    """A non-serializable value nested in a container is rejected."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"items": [1, 2, lambda: None]}

    result = runner.invoke(cli, ["stage1"])
    assert result.exit_code == 1
    assert isinstance(result.exception, TypeError)


def test_pipeline_requires_chain(runner):
    """Pipeline mode requires chain mode."""
    with pytest.raises(ValueError, match="chain mode"):

        @click.group(chain=False, pipeline=True)
        def cli():
            pass


def test_pipeline_tuple_contract(runner):
    """A tuple of types is accepted as the consumer contract."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return [1, 2]

    @cli.command()
    @click.pipeline_input((dict, list))
    def stage2(pipeline_input):
        click.echo(type(pipeline_input).__name__)

    result = runner.invoke(cli, ["stage1", "stage2"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["list"]


def test_pipeline_without_contract(runner):
    """A consumer with no contract accepts any basic container."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"a": 1}

    @cli.command()
    @click.pipeline_input
    def stage2(pipeline_input):
        click.echo(pipeline_input)

    result = runner.invoke(cli, ["stage1", "stage2"])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["{'a': 1}"]


def test_pipeline_reserved_name_rejected(runner):
    """A CLI parameter named ``pipeline_input`` is rejected."""

    @click.group(chain=True, pipeline=True)
    def cli():
        pass

    @cli.command()
    def stage1():
        return {"a": 1}

    @cli.command()
    @click.option("--pipeline-input", is_flag=True)
    @click.pipeline_input(dict)
    def stage2(pipeline_input):
        pass

    result = runner.invoke(cli, ["stage1", "stage2"], standalone_mode=False)
    assert isinstance(result.exception, click.UsageError)


def test_pipeline_empty(runner):
    """A pipeline group with no stages invokes the result callback with []."""

    @click.group(chain=True, pipeline=True, invoke_without_command=True)
    def cli():
        pass

    @cli.result_callback()
    def process(results):
        click.echo(results)

    result = runner.invoke(cli, [])
    assert result.exit_code == 0
    assert result.output.splitlines() == ["[]"]
