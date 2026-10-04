"""Tests for click.MutuallyExclusiveGroup and click.mutually_exclusive."""

import pytest

import click
from click.testing import CliRunner


def make_cli(**group_kwargs):
    """A command with mutually exclusive --token and --profile."""

    @click.command()
    @click.mutually_exclusive("token", "profile", **group_kwargs)
    @click.option("--token", envvar="APP_TOKEN")
    @click.option("--profile")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")

    return cli


def test_conflict_from_command_line():
    """Two members given on the command line conflict; the error names the
    parameters and their sources, but never the values."""
    result = CliRunner().invoke(
        make_cli(), ["--token", "tokvalue", "--profile", "profvalue"]
    )

    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()
    assert "--token" in result.output
    assert "--profile" in result.output
    assert "command line" in result.output
    assert "tokvalue" not in result.output
    assert "profvalue" not in result.output


def test_conflict_between_environment_and_command_line():
    """A token from the environment conflicts with an explicit --profile
    on the command line; both sources are reported."""
    result = CliRunner().invoke(
        make_cli(), ["--profile", "profvalue"], env={"APP_TOKEN": "tokvalue"}
    )

    assert result.exit_code == 2
    assert "--token" in result.output
    assert "environment variable 'APP_TOKEN'" in result.output
    assert "--profile" in result.output
    assert "command line" in result.output
    assert "tokvalue" not in result.output
    assert "profvalue" not in result.output


def test_default_does_not_conflict():
    """A plain default on one member never conflicts with a selected one."""

    @click.command()
    @click.mutually_exclusive("token", "profile")
    @click.option("--token", envvar="APP_TOKEN")
    @click.option("--profile", default="defaultprof")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = CliRunner().invoke(cli, [], env={"APP_TOKEN": "tokvalue"})

    assert result.exit_code == 0
    assert "token=tokvalue profile=defaultprof" in result.output


def test_default_map_counts_as_selection():
    """A value from default_map counts as a selection and conflicts
    with a command line value."""
    result = CliRunner().invoke(
        make_cli(),
        ["--token", "tokvalue"],
        default_map={"profile": "profvalue"},
    )

    assert result.exit_code == 2
    assert "--token" in result.output
    assert "command line" in result.output
    assert "--profile" in result.output
    assert "default map" in result.output
    assert "tokvalue" not in result.output
    assert "profvalue" not in result.output


def test_single_selection_passes():
    """Selecting exactly one member works, from any source."""
    runner = CliRunner()

    result = runner.invoke(make_cli(), ["--token", "tokvalue"])
    assert result.exit_code == 0
    assert "token=tokvalue profile=None" in result.output

    result = runner.invoke(make_cli(), [], env={"APP_TOKEN": "tokvalue"})
    assert result.exit_code == 0
    assert "token=tokvalue profile=None" in result.output

    result = runner.invoke(make_cli(), ["--profile", "profvalue"])
    assert result.exit_code == 0
    assert "token=None profile=profvalue" in result.output


def test_no_selection_passes_when_not_required():
    """An optional group may be left unselected."""
    result = CliRunner().invoke(make_cli(), [])

    assert result.exit_code == 0
    assert "token=None profile=None" in result.output


def test_member_callbacks_not_called_on_conflict():
    """On conflict, neither the member callbacks nor the command callback
    run."""
    calls = []

    def callback(ctx, param, value):
        calls.append(param.name)
        return value

    @click.command()
    @click.mutually_exclusive("token", "profile")
    @click.option("--token", callback=callback)
    @click.option("--profile", callback=callback)
    def cli(token, profile):
        calls.append("cli")  # pragma: no cover

    result = CliRunner().invoke(cli, ["--token", "a", "--profile", "b"])

    assert result.exit_code == 2
    assert calls == []


def test_type_conversion_preserved():
    """Type conversion still applies to the surviving member."""

    @click.command()
    @click.mutually_exclusive("token", "number")
    @click.option("--token")
    @click.option("--number", type=int)
    def cli(token, number):
        click.echo(f"number={number!r}")

    result = CliRunner().invoke(cli, ["--number", "42"])

    assert result.exit_code == 0
    assert "number=42" in result.output

    result = CliRunner().invoke(cli, ["--number", "notanint"])
    assert result.exit_code == 2
    assert "notanint" in result.output  # regular BadParameter error


def test_prompt_when_required_and_nothing_selected():
    """A required group with no selection prompts a promptable member."""

    @click.command()
    @click.mutually_exclusive("token", "profile", required=True)
    @click.option("--token", prompt=True)
    @click.option("--profile")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = CliRunner().invoke(cli, [], input="tokvalue\n")

    assert result.exit_code == 0
    assert "Token" in result.output  # the prompt was shown
    assert "token=tokvalue profile=None" in result.output


def test_no_prompt_when_other_member_selected():
    """The prompt does not appear when another member is already selected."""

    @click.command()
    @click.mutually_exclusive("token", "profile", required=True)
    @click.option("--token", prompt=True)
    @click.option("--profile")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = CliRunner().invoke(cli, ["--profile", "profvalue"])

    assert result.exit_code == 0
    assert "Token:" not in result.output  # no prompt shown
    assert "token=None profile=profvalue" in result.output


def test_required_group_without_selection_fails():
    """A required group with no selection and no promptable member fails."""

    @click.command()
    @click.mutually_exclusive("token", "profile", required=True)
    @click.option("--token")
    @click.option("--profile")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")  # pragma: no cover

    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 2
    assert "--token" in result.output
    assert "--profile" in result.output
    assert "required" in result.output


def test_option_aliases():
    """Members declared with aliases conflict regardless of which alias is
    used, and the error hints at all declared names."""

    @click.command()
    @click.mutually_exclusive("token", "profile")
    @click.option("--token", "-t")
    @click.option("--profile", "-p")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")

    runner = CliRunner()

    result = runner.invoke(cli, ["-t", "tokvalue", "-p", "profvalue"])
    assert result.exit_code == 2
    assert "'--token' / '-t'" in result.output
    assert "'--profile' / '-p'" in result.output
    assert "tokvalue" not in result.output
    assert "profvalue" not in result.output

    result = runner.invoke(cli, ["-t", "tokvalue"])
    assert result.exit_code == 0
    assert "token=tokvalue profile=None" in result.output


def test_group_object_as_decorator():
    """A MutuallyExclusiveGroup instance can be shared and applied as a
    decorator to the callback."""

    group = click.MutuallyExclusiveGroup("token", "profile")

    @click.command()
    @group
    @click.option("--token")
    @click.option("--profile")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = CliRunner().invoke(cli, ["--token", "a", "--profile", "b"])

    assert result.exit_code == 2
    assert "--token" in result.output
    assert "--profile" in result.output


def test_group_applied_to_command_object():
    """The decorator also works above @click.command()."""

    @click.mutually_exclusive("token", "profile")
    @click.command()
    @click.option("--token")
    @click.option("--profile")
    def cli(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = CliRunner().invoke(cli, ["--token", "a", "--profile", "b"])

    assert result.exit_code == 2
    assert "--token" in result.output


def test_unknown_parameter_name():
    """Referencing a parameter that does not exist is an error."""

    @click.command()
    @click.mutually_exclusive("token", "unknown")
    @click.option("--token")
    def cli(token):
        pass  # pragma: no cover

    with pytest.raises(ValueError, match="unknown parameter 'unknown'"):
        CliRunner().invoke(cli, [], catch_exceptions=False)


def test_group_requires_two_members():
    with pytest.raises(ValueError, match="at least two"):
        click.MutuallyExclusiveGroup("token")

    with pytest.raises(TypeError, match="must be parameter names"):
        click.MutuallyExclusiveGroup("token", 42)
