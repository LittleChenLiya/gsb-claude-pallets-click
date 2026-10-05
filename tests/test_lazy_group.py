"""Tests for the immutable-manifest based :class:`click.LazyGroup`.

Acceptance points:

- listing top-level help calls no loader,
- executing a command loads it exactly once,
- concurrent execution of the same command still loads it once and
  shares the constructed command,
- same-thread recursive loaders fail with the full chain,
- a failed load is not cached; the next explicit lookup retries,
- regular ``click.Group`` behavior is unchanged.
"""

from __future__ import annotations

import threading

import pytest

import click
from click import LazyEntry
from click import LazyGroup
from click import LazyManifest
from click.testing import CliRunner


class Loaders:
    """Counts per-name loader calls and constructs named commands."""

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}
        self.lock = threading.Lock()

    def make(self, name: str, fail_times: int = 0, delay=None):
        def loader():
            with self.lock:
                self.calls[name] = self.calls.get(name, 0) + 1
                attempt = self.calls[name]

            if delay is not None:
                delay()

            if attempt <= fail_times:
                raise RuntimeError(f"{name} boom {attempt}")

            @click.command(name)
            def command() -> None:
                click.echo(f"ran {name}")

            return command

        return loader

    def count(self, name: str) -> int:
        return self.calls.get(name, 0)


def make_group(loaders: Loaders, *names: str) -> LazyGroup:
    return LazyGroup(
        name="cli",
        manifest=LazyManifest(
            LazyEntry(name, f"Help for {name}.", loaders.make(name)) for name in names
        ),
    )


def test_help_lists_manifest_without_loading(runner: CliRunner) -> None:
    loaders = Loaders()
    cli = make_group(loaders, "a", "b")

    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0
    assert "a  Help for a." in result.output
    assert "b  Help for b." in result.output
    assert loaders.count("a") == 0
    assert loaders.count("b") == 0


def test_execute_loads_once(runner: CliRunner) -> None:
    loaders = Loaders()
    cli = make_group(loaders, "a", "b")

    result = runner.invoke(cli, ["a"])
    assert result.exit_code == 0
    assert "ran a" in result.output
    assert loaders.count("a") == 1
    assert loaders.count("b") == 0

    # A second invocation (and a help invocation) reuse the command.
    assert runner.invoke(cli, ["a"]).exit_code == 0
    help_result = runner.invoke(cli, ["a", "--help"])
    assert help_result.exit_code == 0
    assert loaders.count("a") == 1


def test_subcommand_help_loads_only_that_command(runner: CliRunner) -> None:
    loaders = Loaders()
    cli = make_group(loaders, "a", "b")

    result = runner.invoke(cli, ["b", "--help"])

    assert result.exit_code == 0
    assert loaders.count("b") == 1
    assert loaders.count("a") == 0


def test_unknown_command_loads_nothing(runner: CliRunner) -> None:
    loaders = Loaders()
    cli = make_group(loaders, "a", "b")

    result = runner.invoke(cli, ["nope"])

    assert result.exit_code == 2
    assert "No such command 'nope'." in result.output
    assert loaders.count("a") == 0
    assert loaders.count("b") == 0


def test_completion_uses_manifest_without_loading() -> None:
    loaders = Loaders()
    cli = make_group(loaders, "alpha", "beta")

    ctx = click.Context(cli)
    items = cli.shell_complete(ctx, "a")

    assert [item.value for item in items] == ["alpha"]
    assert items[0].help == "Help for alpha."
    assert loaders.count("alpha") == 0
    assert loaders.count("beta") == 0


def test_concurrent_same_command_loads_once() -> None:
    loaders = Loaders()
    release = threading.Event()
    entered = threading.Event()

    def loader():
        loaders.calls["a"] = loaders.calls.get("a", 0) + 1
        entered.set()
        release.wait(2)

        @click.command("a")
        def a() -> None:
            click.echo("ran a")

        return a

    cli = LazyGroup(
        name="cli",
        manifest=LazyManifest([LazyEntry("a", "Help for a.", loader)]),
    )
    commands: list[click.Command] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            commands.append(cli.get_command(click.Context(cli), "a"))
        except BaseException as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()

    assert entered.wait(2)
    release.set()
    for thread in threads:
        thread.join()

    assert errors == []
    assert loaders.count("a") == 1
    assert len({id(command) for command in commands}) == 1
    # After the first success, no more loader calls ever happen.
    assert cli.get_command(click.Context(cli), "a") is commands[0]
    assert loaders.count("a") == 1


def test_direct_recursive_loader_reports_chain() -> None:
    group_holder: dict[str, LazyGroup] = {}

    def loader():
        return group_holder["g"].get_command(click.Context(group_holder["g"]), "r")

    group_holder["g"] = LazyGroup(
        manifest=LazyManifest([LazyEntry("r", "recursive", loader)])
    )
    g = group_holder["g"]

    with pytest.raises(click.LazyCommandRecursionError) as exc_info:
        g.get_command(click.Context(g), "r")

    assert exc_info.value.chain == ("r", "r")
    assert "'r' -> 'r'" in str(exc_info.value)
    # The failed attempt leaves no loading state behind.
    assert not g.is_loaded("r")
    with pytest.raises(click.LazyCommandRecursionError):
        g.get_command(click.Context(g), "r")


def test_cross_entry_recursive_loader_reports_chain() -> None:
    holder: dict[str, LazyGroup] = {}

    def load_p():
        return holder["g"].get_command(click.Context(holder["g"]), "q")

    def load_q():
        return holder["g"].get_command(click.Context(holder["g"]), "p")

    holder["g"] = LazyGroup(
        manifest=LazyManifest(
            [LazyEntry("p", "p", load_p), LazyEntry("q", "q", load_q)]
        )
    )
    g = holder["g"]

    with pytest.raises(click.LazyCommandRecursionError) as exc_info:
        g.get_command(click.Context(g), "p")

    assert exc_info.value.chain == ("p", "q", "p")


def test_failed_load_is_retried_on_next_lookup(runner: CliRunner) -> None:
    loaders = Loaders()
    cli = LazyGroup(
        name="cli",
        manifest=LazyManifest(
            [LazyEntry("a", "Help for a.", loaders.make("a", fail_times=1))]
        ),
    )

    first = runner.invoke(cli, ["a"])
    assert first.exit_code == 1
    assert "Could not lazily load command 'a'" in first.output
    assert loaders.count("a") == 1
    assert not cli.is_loaded("a")

    # The explicit next lookup clears the failed attempt and retries.
    second = runner.invoke(cli, ["a"])
    assert second.exit_code == 0
    assert "ran a" in second.output
    assert loaders.count("a") == 2

    third = runner.invoke(cli, ["a"])
    assert third.exit_code == 0
    assert loaders.count("a") == 2


def test_concurrent_failure_is_shared_then_retried() -> None:
    release = threading.Event()
    entered = threading.Event()
    calls = {"n": 0}

    def loader():
        calls["n"] += 1
        entered.set()
        release.wait(2)
        raise RuntimeError("boom")

    g = LazyGroup(manifest=LazyManifest([LazyEntry("a", "a", loader)]))
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            g.get_command(click.Context(g), "a")
        except BaseException as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    assert entered.wait(2)
    release.set()
    for thread in threads:
        thread.join()

    assert calls["n"] == 1
    assert len(errors) == 5
    assert all(isinstance(error, click.LazyLoadError) for error in errors)
    assert not g.is_loaded("a")

    # A fresh lookup retries the loader.
    calls["n"] = 0

    @click.command("a")
    def a() -> None:
        click.echo("ok")

    # Swap behavior by reusing a new group is unnecessary: instead verify
    # a retry invokes the loader again and surfaces its failure.
    with pytest.raises(click.LazyLoadError):
        g.get_command(click.Context(g), "a")
    assert calls["n"] == 1


def test_click_exception_from_loader_is_not_wrapped(runner: CliRunner) -> None:
    message = "loader usage problem"

    def loader():
        raise click.UsageError(message)

    g = LazyGroup(name="cli", manifest=LazyManifest([LazyEntry("a", "a", loader)]))
    result = runner.invoke(g, ["a"])
    assert result.exit_code == 2
    assert message in result.output
    assert "Could not lazily load" not in result.output


def test_loader_returning_non_command_fails_and_retries() -> None:
    g = LazyGroup(manifest=LazyManifest([LazyEntry("a", "a", lambda: "not a command")]))
    with pytest.raises(click.LazyLoadError) as exc_info:
        g.get_command(click.Context(g), "a")
    assert "must return a Command instance" in str(exc_info.value)
    assert not g.is_loaded("a")


def test_manifest_is_immutable_and_validated() -> None:
    def loader():
        return click.Command("a")

    manifest = LazyManifest([LazyEntry("a", "help", loader)])

    with pytest.raises(TypeError):
        manifest["a"] = LazyEntry("b", "", loader)  # type: ignore[index]

    with pytest.raises(ValueError, match="Duplicate"):
        LazyManifest([LazyEntry("a", "", loader), LazyEntry("a", "", loader)])
    with pytest.raises(ValueError, match="non-empty"):
        LazyManifest([LazyEntry("", "", loader)])
    with pytest.raises(TypeError, match="callable"):
        LazyManifest([LazyEntry("a", "", 123)])  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        LazyManifest([("a", "", loader, False, "extra")])

    assert manifest.ordered_names() == ("a",)
    assert list(manifest) == ["a"]
    assert len(manifest) == 1


def test_lazy_group_rejects_mutation() -> None:
    g = LazyGroup(manifest=LazyManifest([]))

    with pytest.raises(TypeError, match="immutable"):
        g.add_command(click.Command("a"))
    with pytest.raises(TypeError, match="manifest"):
        LazyGroup(commands={"a": click.Command("a")})

    # The exposed command view only reflects successful loads.
    assert dict(g.commands) == {}


def test_to_info_dict_does_not_load() -> None:
    loaders = Loaders()
    cli = make_group(loaders, "a", "b")
    ctx = click.Context(cli)

    info = cli.to_info_dict(ctx)

    assert set(info["commands"]) == {"a", "b"}
    assert info["commands"]["a"]["short_help"] == "Help for a."
    assert info["commands"]["a"]["loaded"] is False
    assert loaders.count("a") == 0
    assert loaders.count("b") == 0


def test_regular_group_is_unchanged(runner: CliRunner) -> None:
    @click.command()
    def a() -> None:
        click.echo("plain a")

    cli = click.Group(name="cli", commands=[a])

    result = runner.invoke(cli, ["a"])
    assert result.exit_code == 0
    assert "plain a" in result.output

    @click.command()
    def b() -> None:
        click.echo("plain b")

    cli.add_command(b)
    assert runner.invoke(cli, ["b"]).exit_code == 0
    assert set(cli.list_commands(click.Context(cli))) == {"a", "b"}


def test_loaders_run_outside_lock_other_commands_progress() -> None:
    release_a = threading.Event()
    a_started = threading.Event()

    def load_a():
        a_started.set()
        release_a.wait(2)
        return click.Command("a", callback=lambda: None)

    def load_b():
        return click.Command("b", callback=lambda: None)

    cli = LazyGroup(
        manifest=LazyManifest(
            [LazyEntry("a", "a", load_a), LazyEntry("b", "b", load_b)]
        )
    )
    b_result: list[click.Command] = []

    def load_b_worker():
        b_result.append(cli.get_command(click.Context(cli), "b"))

    thread_a = threading.Thread(target=cli.get_command, args=(click.Context(cli), "a"))
    thread_a.start()
    assert a_started.wait(2)

    # While "a" is mid-load, loading a different command must not block.
    thread_b = threading.Thread(target=load_b_worker)
    thread_b.start()
    thread_b.join(2)
    assert not thread_b.is_alive(), "loading 'b' blocked behind loader of 'a'"
    assert b_result and b_result[0].name == "b"

    release_a.set()
    thread_a.join(2)


def test_shell_completion_uses_manifest_end_to_end() -> None:
    from click.shell_completion import ShellComplete

    loaders = Loaders()
    cli = make_group(loaders, "alpha", "beta")
    comp = ShellComplete(cli, {}, cli.name, "_CLICK_COMPLETE")

    items = comp.get_completions([], "")
    assert [item.value for item in items] == ["alpha", "beta"]
    assert items[0].help == "Help for alpha."
    assert loaders.count("alpha") == 0
    assert loaders.count("beta") == 0

    items = comp.get_completions([], "al")
    assert [item.value for item in items] == ["alpha"]
    assert loaders.count("alpha") == 0
