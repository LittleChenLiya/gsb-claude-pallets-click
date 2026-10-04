"""Tests for explicit mutually exclusive parameter groups."""

import click
from click.testing import CliRunner


def test_options_are_mutually_exclusive_on_command_line(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--token", "tokval", "--profile", "profval"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()
    assert "--token" in result.output
    assert "--profile" in result.output
    error = result.output.split("Error:")[1]
    assert "tokval" not in error
    assert "profval" not in error


def test_environment_and_command_line_conflict(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", envvar="TOKEN", group=group)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--profile", "work"], env={"TOKEN": "secret"})
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()
    assert "environment" in result.output
    assert "command line" in result.output


def test_default_value_does_not_conflict(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", envvar="TOKEN", group=group)
    @click.option("--profile", default="default_profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, [], env={"TOKEN": "secret"})
    assert result.exit_code == 0
    assert "token=secret" in result.output
    assert "profile=default_profile" in result.output


def test_default_map_is_a_selection(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(
        cmd, ["--profile", "work"], default_map={"token": "mapped"}
    )
    assert result.exit_code == 2
    assert "default map" in result.output
    assert "command line" in result.output


def test_callbacks_not_called_on_conflict(runner):
    calls = []

    def token_cb(ctx, param, value):
        calls.append(("token", value))
        return value

    def profile_cb(ctx, param, value):
        calls.append(("profile", value))
        return value

    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", envvar="TOKEN", group=group, callback=token_cb)
    @click.option("--profile", group=group, callback=profile_cb)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--profile", "work"], env={"TOKEN": "secret"})
    assert result.exit_code == 2
    assert calls == []


def test_single_selection_is_fine(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--token", "a"])
    assert result.exit_code == 0
    assert "token=a" in result.output
    assert "profile=None" in result.output


def test_no_selection_non_required_group(runner):
    group = click.MutuallyExclusiveGroup(name="auth", required=False)

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, [])
    assert result.exit_code == 0
    assert "token=None" in result.output
    assert "profile=None" in result.output


def test_required_group_prompts_when_nothing_selected(runner):
    group = click.MutuallyExclusiveGroup(name="auth", required=True)

    @click.command()
    @click.option("--token", group=group, prompt=True)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, [], input="mytoken\n")
    assert result.exit_code == 0
    assert "Token:" in result.output
    assert "token=mytoken" in result.output


def test_required_group_does_not_prompt_when_selected(runner):
    group = click.MutuallyExclusiveGroup(name="auth", required=True)

    @click.command()
    @click.option("--token", group=group, prompt=True)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--profile", "work"])
    assert result.exit_code == 0
    assert "Token:" not in result.output
    assert "profile=work" in result.output


def test_required_group_conflict_does_not_prompt(runner):
    group = click.MutuallyExclusiveGroup(name="auth", required=True)

    @click.command()
    @click.option("--token", group=group, prompt=True)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--token", "tokval", "--profile", "profval"])
    assert result.exit_code == 2
    assert "Token:" not in result.output
    assert "mutually exclusive" in result.output.lower()


def test_required_group_without_prompt_or_selection_fails(runner):
    group = click.MutuallyExclusiveGroup(name="auth", required=True)

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, [])
    assert result.exit_code == 2
    assert "requires one of" in result.output


def test_parameter_alias_string_reference(runner):
    @click.command()
    @click.option("--token", group="auth")
    @click.option("--profile", group="auth")
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--token", "tokval", "--profile", "profval"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()
    assert "auth" in result.output


def test_mixed_object_and_string_references(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group="auth")
    def cmd(token, profile):
        click.echo(f"token={token} profile={profile}")

    result = runner.invoke(cmd, ["--token", "tokval", "--profile", "profval"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()


def test_type_conversion_still_runs(runner):
    group = click.MutuallyExclusiveGroup(name="nums")

    @click.command()
    @click.option("--a", type=int, group=group)
    @click.option("--b", type=int, group=group)
    def cmd(a, b):
        click.echo(f"a={a!r} b={b!r}")

    result = runner.invoke(cmd, ["--a", "1", "--b", "2"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output.lower()

    result = runner.invoke(cmd, ["--a", "notanint"])
    assert result.exit_code == 2
    assert "is not a valid integer" in result.output


def test_three_way_conflict(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group=group)
    @click.option("--cert", group=group)
    def cmd(token, profile, cert):
        click.echo(f"token={token} profile={profile} cert={cert}")

    result = runner.invoke(cmd, ["--token", "a", "--profile", "b", "--cert", "c"])
    assert result.exit_code == 2
    assert result.output.count("--") >= 3


def test_multiple_groups_are_independent(runner):
    auth = click.MutuallyExclusiveGroup(name="auth")
    fmt = click.MutuallyExclusiveGroup(name="fmt")

    @click.command()
    @click.option("--token", group=auth)
    @click.option("--profile", group=auth)
    @click.option("--json", "output_fmt", flag_value="json", group=fmt)
    @click.option("--yaml", "output_fmt", flag_value="yaml", group=fmt)
    def cmd(token, profile, output_fmt):
        click.echo(f"token={token} profile={profile} fmt={output_fmt}")

    result = runner.invoke(cmd, ["--token", "a", "--json"])
    assert result.exit_code == 0
    assert "token=a" in result.output
    assert "fmt=json" in result.output

    result = runner.invoke(cmd, ["--token", "a", "--profile", "b", "--json"])
    assert result.exit_code == 2
    assert "auth" in result.output
    error = result.output.split("Error:")[1]
    assert "fmt" not in error


def test_group_members_property(runner):
    group = click.MutuallyExclusiveGroup(name="auth")

    @click.command()
    @click.option("--token", group=group)
    @click.option("--profile", group=group)
    def cmd(token, profile):
        pass

    names = [p.name for p in group.members]
    assert names == ["profile", "token"]
