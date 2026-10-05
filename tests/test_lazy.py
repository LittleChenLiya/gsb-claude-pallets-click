import threading

import click


def test_lazy_group_help_does_not_load(runner):
    load_counts = {"a": 0, "b": 0}

    def load_a():
        load_counts["a"] += 1
        return click.Command("a", help="Command A full help.")

    def load_b():
        load_counts["b"] += 1
        return click.Command("b", help="Command B full help.")

    manifest = click.LazyManifest(
        [
            click.LazyCommandEntry("a", "Short help for A", load_a),
            click.LazyCommandEntry("b", "Short help for B", load_b),
        ]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Commands:" in result.output
    assert "a" in result.output
    assert "b" in result.output
    assert "Short help for A" in result.output
    assert "Short help for B" in result.output
    assert load_counts == {"a": 0, "b": 0}


def test_lazy_group_completion_does_not_load(runner):
    load_counts = {"a": 0, "b": 0}

    def load_a():
        load_counts["a"] += 1
        return click.Command("a")

    def load_b():
        load_counts["b"] += 1
        return click.Command("b")

    manifest = click.LazyManifest(
        [
            click.LazyCommandEntry("a", "Short help for A", load_a),
            click.LazyCommandEntry("b", "Short help for B", load_b),
        ]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    result = runner.invoke(
        cli,
        env={
            "_CLI_COMPLETE": "bash_complete",
            "COMP_WORDS": "cli",
            "COMP_CWORD": "1",
        },
        prog_name="cli",
    )
    assert result.exit_code == 0
    assert "plain,a" in result.output
    assert "plain,b" in result.output
    assert load_counts == {"a": 0, "b": 0}


def test_lazy_group_execute_loads_once(runner):
    load_counts = {"a": 0, "b": 0}

    def load_a():
        load_counts["a"] += 1
        return click.Command("a", callback=lambda: click.echo("a running"))

    def load_b():
        load_counts["b"] += 1
        return click.Command("b", callback=lambda: click.echo("b running"))

    manifest = click.LazyManifest(
        [
            click.LazyCommandEntry("a", "Short help for A", load_a),
            click.LazyCommandEntry("b", "Short help for B", load_b),
        ]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    result = runner.invoke(cli, ["a"])
    assert result.exit_code == 0
    assert "a running" in result.output
    assert load_counts == {"a": 1, "b": 0}

    result = runner.invoke(cli, ["a"])
    assert result.exit_code == 0
    assert load_counts == {"a": 1, "b": 0}


def test_lazy_group_help_of_command_loads_once(runner):
    load_counts = {"a": 0}

    def load_a():
        load_counts["a"] += 1
        return click.Command("a", help="Command A full help.")

    manifest = click.LazyManifest(
        [click.LazyCommandEntry("a", "Short help for A", load_a)]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    result = runner.invoke(cli, ["a", "--help"])
    assert result.exit_code == 0
    assert "Command A full help." in result.output
    assert load_counts == {"a": 1}


def test_lazy_group_concurrent_loads_once():
    load_count = [0]
    load_lock = threading.Lock()

    def load_slow():
        with load_lock:
            load_count[0] += 1
        import time

        time.sleep(0.05)
        return click.Command("slow")

    manifest = click.LazyManifest(
        [click.LazyCommandEntry("slow", "Slow command", load_slow)]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    results = []
    errors = []

    def worker():
        try:
            results.append(cli.get_command(None, "slow"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert load_count[0] == 1
    assert len(results) == 10
    assert not errors
    assert all(r is results[0] for r in results)


def test_lazy_group_recursive_load_reports_chain():
    def load_recursive():
        return cli.get_command(None, "recursive")

    manifest = click.LazyManifest(
        [click.LazyCommandEntry("recursive", "Recursive", load_recursive)]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    try:
        cli.get_command(None, "recursive")
        raise AssertionError("Should have raised RuntimeError")
    except RuntimeError as e:
        assert "Recursive load detected" in str(e)
        assert "recursive -> recursive" in str(e)


def test_lazy_group_failure_cached_and_retried():
    attempts = [0]

    def load_fail():
        attempts[0] += 1
        if attempts[0] <= 2:
            raise ValueError(f"load failed (attempt {attempts[0]})")
        return click.Command("fail")

    manifest = click.LazyManifest(
        [click.LazyCommandEntry("fail", "Fail command", load_fail)]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    try:
        cli.get_command(None, "fail")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "attempt 1" in str(e)

    try:
        cli.get_command(None, "fail")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "attempt 1" in str(e)

    try:
        cli.retry_load("fail")
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "attempt 2" in str(e)

    cmd = cli.retry_load("fail")
    assert cmd is not None
    cmd = cli.get_command(None, "fail")
    assert cmd is not None


def test_lazy_group_hidden_command_not_shown(runner):
    load_count = [0]

    def load_hidden():
        load_count[0] += 1
        return click.Command("hidden", hidden=True)

    manifest = click.LazyManifest(
        [
            click.LazyCommandEntry(
                "visible", "Visible", lambda: click.Command("visible")
            ),
            click.LazyCommandEntry("hidden", "Hidden", load_hidden, hidden=True),
        ]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    result = runner.invoke(cli, ["--help"])
    assert "visible" in result.output
    assert "hidden" not in result.output
    assert load_count[0] == 0


def test_lazy_group_unknown_command_returns_none():
    """A command not in the manifest returns None without deadlocking."""
    manifest = click.LazyManifest(
        [click.LazyCommandEntry("a", "A", lambda: click.Command("a"))]
    )
    cli = click.LazyGroup("cli", manifest=manifest)

    # Unknown command returns None.
    assert cli.get_command(None, "unknown") is None

    # Calling again should not deadlock (loading state cleaned up).
    assert cli.get_command(None, "unknown") is None


def test_lazy_group_manifest_from_dict():
    cli = click.LazyGroup(
        "cli",
        manifest={
            "a": ("Short help for A", lambda: click.Command("a")),
            "b": ("Short help for B", lambda: click.Command("b"), True),
        },
    )
    assert cli.list_commands(None) == ["a", "b"]
    cmd = cli.get_command(None, "a")
    assert cmd is not None
    assert cmd.name == "a"
