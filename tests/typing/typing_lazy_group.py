"""Type-check examples for the immutable-manifest LazyGroup."""

from __future__ import annotations

from collections.abc import Mapping

from typing_extensions import assert_type

import click
from click.shell_completion import CompletionItem


def load_a() -> click.Command:
    return click.Command("a")


manifest = click.LazyManifest(
    [
        click.LazyEntry("a", "Short help for a.", load_a),
        ("b", "Short help for b.", load_a, True),
    ]
)
assert_type(manifest["a"], click.LazyEntry)
assert_type(manifest.ordered_names(), tuple[str, ...])

cli = click.LazyGroup(name="cli", manifest=manifest)
assert_type(cli, click.LazyGroup)
assert_type(cli.manifest, click.LazyManifest)
assert_type(cli.loaded_commands, Mapping[str, click.Command])

ctx = click.Context(cli)
assert_type(cli.list_commands(ctx), list[str])
assert_type(cli.get_command(ctx, "a"), click.Command | None)
assert_type(cli.shell_complete(ctx, ""), list[CompletionItem])

# Building one from an iterable of tuples works too.
click.LazyGroup(manifest=[("a", "help", load_a)])
