"""Lazy groups based on an immutable manifest.

A :class:`LazyGroup` provides help and completion using only command names and
short descriptions from a manifest, without importing all command modules.
Commands are loaded on demand when selected or when their help is requested.
"""

from __future__ import annotations

import collections.abc as cabc
import inspect
import threading
import typing as t
from gettext import gettext as _

from .core import Command
from .core import Group
from .exceptions import NoSuchCommand
from .parser import _split_opt
from .utils import make_str

if t.TYPE_CHECKING:
    from .core import Context
    from .formatting import HelpFormatter

__all__ = ["LazyCommandEntry", "LazyManifest", "LazyGroup"]


class LazyCommandEntry:
    """A single entry in a :class:`LazyManifest`.

    Attributes:
        name: The command name.
        short_help: A short description used in help and completion.
        loader: A callable that returns a :class:`Command`.
        hidden: Whether the command is hidden from help and completion.
    """

    __slots__ = ("name", "short_help", "loader", "hidden")

    def __init__(
        self,
        name: str,
        short_help: str,
        loader: t.Callable[[], Command],
        hidden: bool = False,
    ) -> None:
        self.name = name
        self.short_help = short_help
        self.loader = loader
        self.hidden = hidden

    def __repr__(self) -> str:
        return f"<LazyCommandEntry {self.name!r}>"


class LazyManifest:
    """An immutable manifest of lazy commands.

    The manifest maps command names to their short descriptions and loaders.
    It is used by :class:`LazyGroup` to provide help and completion without
    importing all command modules.

    :param entries: A list of :class:`LazyCommandEntry` objects, or a mapping
        from command name to a ``(short_help, loader)`` or
        ``(short_help, loader, hidden)`` tuple.
    """

    def __init__(
        self,
        entries: (
            cabc.Iterable[LazyCommandEntry]
            | cabc.Mapping[
                str,
                tuple[str, t.Callable[[], Command]]
                | tuple[str, t.Callable[[], Command], bool],
            ]
        ),
    ) -> None:
        self._entries: dict[str, LazyCommandEntry] = {}

        if isinstance(entries, cabc.Mapping):
            for name, value in entries.items():
                if not isinstance(value, tuple):
                    raise TypeError(
                        f"Invalid manifest entry for {name!r}: expected a tuple, "
                        f"got {type(value).__name__}"
                    )
                if len(value) == 2:
                    short_help, loader = value
                    hidden = False
                elif len(value) == 3:
                    short_help, loader, hidden = value
                else:
                    raise ValueError(
                        f"Invalid manifest entry for {name!r}: expected 2 or 3 "
                        f"elements, got {len(value)}"
                    )
                self._entries[name] = LazyCommandEntry(name, short_help, loader, hidden)
        else:
            for entry in entries:
                if not isinstance(entry, LazyCommandEntry):
                    raise TypeError(
                        f"Invalid manifest entry: expected LazyCommandEntry, "
                        f"got {type(entry).__name__}"
                    )
                self._entries[entry.name] = entry

    def names(self) -> list[str]:
        """Return all command names in the manifest, sorted."""
        return sorted(self._entries)

    def get(self, name: str) -> LazyCommandEntry | None:
        """Return the entry for the given name, or ``None`` if not found."""
        return self._entries.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __repr__(self) -> str:
        return f"<LazyManifest {list(self._entries)!r}>"


class LazyGroup(Group):
    """A group that loads subcommands on demand from an immutable manifest.

    The manifest provides command names and short descriptions for help and
    completion without importing all command modules. Commands are loaded
    only when selected or when their help is requested.

    Loaded commands are cached and shared. A failed load is cached and raised
    on subsequent access until explicitly retried with :meth:`retry_load`.

    :param name: The name of the group command.
    :param manifest: A :class:`LazyManifest`, list of :class:`LazyCommandEntry`,
        or mapping from name to ``(short_help, loader)`` tuple.
    :param load_timeout: Maximum time in seconds to wait for another thread
        to finish loading the same command. Defaults to 30.
    :param kwargs: Other arguments passed to :class:`Group`.
    """

    def __init__(
        self,
        name: str | None = None,
        manifest: (
            LazyManifest
            | cabc.Iterable[LazyCommandEntry]
            | cabc.Mapping[
                str,
                tuple[str, t.Callable[[], Command]]
                | tuple[str, t.Callable[[], Command], bool],
            ]
            | None
        ) = None,
        load_timeout: float = 30.0,
        **kwargs: t.Any,
    ) -> None:
        super().__init__(name, **kwargs)

        if manifest is None:
            manifest = LazyManifest([])
        elif not isinstance(manifest, LazyManifest):
            manifest = LazyManifest(manifest)

        self._manifest = manifest
        self._load_timeout = load_timeout

        #: Successfully loaded commands, keyed by name.
        self._cache: dict[str, Command] = {}
        #: Cached exceptions from failed loads, keyed by name.
        self._failed: dict[str, BaseException] = {}
        #: Events for commands currently being loaded, keyed by name.
        self._loading: dict[str, threading.Event] = {}
        #: Protects the cache, failed, and loading dicts.
        self._lock = threading.Lock()
        #: Thread-local stack of names currently being loaded.
        self._local = threading.local()

    def _get_loading_stack(self) -> list[str]:
        if not hasattr(self._local, "stack"):
            self._local.stack = []
        stack: list[str] = self._local.stack
        return stack

    def list_commands(self, ctx: Context) -> list[str]:
        """Return all subcommand names from the manifest, sorted."""
        return self._manifest.names()

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:
        """Load and return the command for the given name.

        The command is loaded from the manifest's loader on first access and
        cached for subsequent access. A failed load is cached and raised on
        subsequent access until :meth:`retry_load` is called.

        :param ctx: Invocation context (unused, kept for compatibility).
        :param cmd_name: The name of the command to load.
        """
        # Fast path: already cached.
        if cmd_name in self._cache:
            return self._cache[cmd_name]

        stack = self._get_loading_stack()

        with self._lock:
            # Double-check after acquiring the lock.
            if cmd_name in self._cache:
                return self._cache[cmd_name]

            # Raise cached failure without retrying.
            if cmd_name in self._failed:
                raise self._failed[cmd_name]

            # Detect recursive loading on the same thread.
            if cmd_name in stack:
                chain = " -> ".join(stack + [cmd_name])
                raise RuntimeError(
                    f"Recursive load detected for command {cmd_name!r}: {chain}"
                )

            # If another thread is loading the same command, wait for it.
            if cmd_name in self._loading:
                event = self._loading[cmd_name]
                is_loader = False
            else:
                # This thread is the loader.
                event = threading.Event()
                self._loading[cmd_name] = event
                is_loader = True
                stack.append(cmd_name)

        if not is_loader:
            # Wait for the loading thread to finish.
            if not event.wait(self._load_timeout):
                raise RuntimeError(
                    f"Timed out waiting for command {cmd_name!r} to load "
                    f"(possible recursive load across threads)"
                )
            # Re-check cache after waiting.
            with self._lock:
                if cmd_name in self._cache:
                    return self._cache[cmd_name]
                if cmd_name in self._failed:
                    raise self._failed[cmd_name]
                return None

        # Call the loader outside the lock so other commands can load
        # concurrently.
        try:
            entry = self._manifest.get(cmd_name)
            cmd = entry.loader() if entry is not None else None
        except BaseException as e:
            with self._lock:
                self._failed[cmd_name] = e
                del self._loading[cmd_name]
                stack.pop()
            event.set()
            raise
        else:
            with self._lock:
                if cmd is not None:
                    self._cache[cmd_name] = cmd
                del self._loading[cmd_name]
                stack.pop()
            event.set()
            return cmd

    def retry_load(self, cmd_name: str) -> Command | None:
        """Clear a cached failure and retry loading the command.

        This is the explicit retry mechanism for failed loads. It clears the
        cached failure (and any cached success) and reloads the command.

        :param cmd_name: The name of the command to retry.
        :return: The loaded command, or ``None`` if not found.
        :raises BaseException: If the retry fails.
        """
        with self._lock:
            self._failed.pop(cmd_name, None)
            self._cache.pop(cmd_name, None)
        return self.get_command(None, cmd_name)  # type: ignore[arg-type]

    def resolve_command(
        self, ctx: Context, args: list[str]
    ) -> tuple[str | None, Command | None, list[str]]:
        cmd_name = make_str(args[0])

        # Get the command (loads it if necessary).
        cmd = self.get_command(ctx, cmd_name)

        # If we can't find the command but there is a normalization
        # function available, we try with that one.
        if cmd is None and ctx.token_normalize_func is not None:
            cmd_name = ctx.token_normalize_func(cmd_name)
            cmd = self.get_command(ctx, cmd_name)

        # If we don't find the command we want to show an error message
        # to the user that it was not provided.
        if cmd is None and not ctx.resilient_parsing:
            if _split_opt(cmd_name)[0]:
                self.parse_args(ctx, args)
            raise NoSuchCommand(cmd_name, possibilities=self._manifest.names(), ctx=ctx)
        return cmd_name if cmd else None, cmd, args[1:]

    def format_commands(self, ctx: Context, formatter: HelpFormatter) -> None:
        """Format the command list using the manifest, without loading commands."""
        commands = []
        for name in self.list_commands(ctx):
            entry = self._manifest.get(name)
            if entry is None or entry.hidden:
                continue
            commands.append((name, entry))

        if commands:
            rows = []
            for name, entry in commands:
                short_help = inspect.cleandoc(entry.short_help)
                rows.append((name, short_help))
            if rows:
                with formatter.section(_("Commands")):
                    formatter.write_dl(rows)

    def shell_complete(self, ctx: Context, incomplete: str) -> list[t.Any]:
        """Return completions using the manifest, without loading commands."""
        from click.shell_completion import CompletionItem

        results = [
            CompletionItem(name, help=inspect.cleandoc(entry.short_help))
            for name in self.list_commands(ctx)
            if name.startswith(incomplete)
            for entry in [self._manifest.get(name)]
            if entry is not None and not entry.hidden
        ]
        # Call Command.shell_complete (not Group.shell_complete) to avoid
        # loading commands via _complete_visible_commands.
        results.extend(Command.shell_complete(self, ctx, incomplete))
        return results

    def to_info_dict(self, ctx: Context) -> dict[str, t.Any]:
        """Return info from the manifest, without loading commands."""
        info_dict = Command.to_info_dict(self, ctx)
        commands = {}
        for name in self.list_commands(ctx):
            entry = self._manifest.get(name)
            if entry is None:
                continue
            commands[name] = {
                "name": entry.name,
                "short_help": entry.short_help,
                "hidden": entry.hidden,
            }
        info_dict.update(commands=commands, chain=self.chain)
        return info_dict
