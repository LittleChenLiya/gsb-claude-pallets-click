"""Lazy groups backed by an immutable manifest.

A :class:`LazyGroup` knows the names and short descriptions of its
subcommands up front, but does not import or construct their command
modules until a subcommand is actually selected. This makes top-level
``--help`` output and shell completion cheap: they use the manifest
metadata only.

Command loaders are called at most once per successful load. The loaded
:class:`~click.Command` is cached and shared by every later lookup in
every thread. If a loader fails, nothing is cached: the failure is
reported to all threads that were waiting for that load and the next
explicit lookup clears the attempt and retries the loader.

Only :class:`LazyGroup` gains this behavior; the regular
:class:`~click.Group` is unchanged.

.. versionadded:: 8.6.0
"""

from __future__ import annotations

import collections.abc as cabc
import threading
import typing as t
from gettext import gettext as _
from types import MappingProxyType

from .core import Command
from .core import Group
from .exceptions import ClickException
from .formatting import HelpFormatter
from .shell_completion import CompletionItem
from .utils import _make_default_short_help
from .utils import make_str

if t.TYPE_CHECKING:
    from .core import Context


# Callable returning (or promising to return) the command for one entry.
LoaderType = t.Callable[[], Command]


class LazyLoadError(ClickException):
    """Raised when a lazy command loader raises something other than a
    :class:`~click.exceptions.ClickException`.

    The original exception is available as ``__cause__``.

    .. versionadded:: 8.6.0
    """

    exit_code: t.ClassVar[int] = 1

    #: Name of the manifest entry whose loader failed.
    command_name: str

    def __init__(self, command_name: str, error: BaseException) -> None:
        self.command_name = command_name
        super().__init__(f"Could not lazily load command {command_name!r}: {error}")


class LazyCommandRecursionError(LazyLoadError):
    """Raised when loading a command triggers loading the same entry
    again on the same thread.

    This is a subclass of :class:`LazyLoadError`. :attr:`chain` lists the
    entries that were being loaded, in order, ending with the entry that
    would be loaded again.

    .. versionadded:: 8.6.0
    """

    #: Entries being loaded on the same thread, from the first to the
    #: entry that repeated.
    chain: t.Final[tuple[str, ...]]

    def __init__(self, chain: cabc.Sequence[str]) -> None:
        self.chain = tuple(chain)
        last = self.chain[-1]
        path = " -> ".join(repr(name) for name in self.chain)
        ClickException.__init__(
            self,
            f"Recursive lazy load detected for command {last!r}: {path}",
        )
        self.command_name = last


class LazyEntry(t.NamedTuple):
    """One entry of a :class:`LazyManifest`.

    :param name: Subcommand name used on the command line.
    :param short_help: One-line description shown in the parent group help
        and in shell completion, without loading the command.
    :param loader: Zero-argument callable that imports and returns the
        :class:`~click.Command`. It is called only when the command is
        selected (including ``name --help``).
    :param hidden: Hide the command from help listings and completion.

    .. versionadded:: 8.6.0
    """

    name: str
    short_help: str
    loader: LoaderType
    hidden: bool = False


# Accepted input for a single manifest entry: a full entry, or a
# (name, short_help, loader[, hidden]) tuple.
EntryInput = LazyEntry | tuple[t.Any, ...]


class LazyManifest(cabc.Mapping[str, LazyEntry]):
    """Immutable map of command names to :class:`LazyEntry`.

    The manifest is built once from manifest entries (or
    ``(name, short_help, loader[, hidden])`` tuples) and cannot change
    afterwards. Declaration order is preserved and exposed through
    :meth:`ordered_names`.

    :param entries: Entries, one per subcommand.
    :raises TypeError: If an entry has the wrong shape or the loader is
        not callable.
    :raises ValueError: If a name is duplicated or empty.

    .. versionadded:: 8.6.0
    """

    __slots__ = ("_entries",)
    _entries: dict[str, LazyEntry]

    def __init__(self, entries: cabc.Iterable[EntryInput]) -> None:
        result: dict[str, LazyEntry] = {}

        for raw in entries:
            if isinstance(raw, LazyEntry):
                entry = raw
            elif isinstance(raw, tuple):
                try:
                    entry = LazyEntry(*raw)
                except TypeError as e:
                    raise TypeError(f"Invalid lazy manifest entry {raw!r}: {e}") from e
            else:
                raise TypeError(
                    "Lazy manifest entries must be LazyEntry objects or"
                    f" tuples, got {type(raw).__name__}."
                )

            if not isinstance(entry.name, str) or not entry.name:
                raise ValueError(
                    "Lazy manifest entry name must be a non-empty string,"
                    f" got {entry.name!r}."
                )

            if not callable(entry.loader):
                raise TypeError(
                    f"Loader for lazy command {entry.name!r} must be callable."
                )

            if entry.name in result:
                raise ValueError(f"Duplicate lazy command name {entry.name!r}.")

            result[entry.name] = entry

        object.__setattr__(self, "_entries", result)

    def __getitem__(self, name: str) -> LazyEntry:
        return self._entries[name]

    def __iter__(self) -> cabc.Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def ordered_names(self) -> tuple[str, ...]:
        """Entry names in the order they were declared."""
        return tuple(self._entries)

    def __repr__(self) -> str:
        names = ", ".join(self._entries)
        return f"<LazyManifest [{names}]>"


class _LoadState:
    """One in-progress load, shared by every thread waiting on it."""

    __slots__ = ("event", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.error: BaseException | None = None


class LazyGroup(Group):
    """A :class:`~click.Group` whose subcommands are declared in an
    immutable :class:`LazyManifest`.

    Top-level help and shell completion use only each entry's name and
    short help; no command module is imported. A command's ``loader`` is
    called only when the command is selected for execution or for its own
    help page.

    Loading semantics:

    -   A successful load is cached and shared across all threads; the
        loader never runs twice for the same name.
    -   Concurrent lookups of the same name block on the single load;
        they never construct two commands.
    -   A failed load is not cached. All concurrent waiters receive the
        failure, and the next explicit lookup retries with a fresh state.
    -   If a loader for an entry triggers loading another entry whose
        loader loops back (on the same thread), a
        :class:`LazyCommandRecursionError` reports the full chain.
    -   The user loader always runs outside the internal lock.

    The manifest cannot be mutated, so :meth:`add_command` is disabled.
    Regular :class:`~click.Group` behavior is not affected.

    :param name: The name of the group command.
    :param manifest: The command manifest, or an iterable accepted by
        :class:`LazyManifest`.
    :param kwargs: Other arguments passed to :class:`~click.Group`.
        ``commands`` may not be given.

    .. versionadded:: 8.6.0
    """

    def __init__(
        self,
        name: str | None = None,
        manifest: LazyManifest | cabc.Iterable[EntryInput] | None = None,
        **kwargs: t.Any,
    ) -> None:
        if "commands" in kwargs:
            raise TypeError(
                "LazyGroup does not accept 'commands'; pass a 'manifest' instead."
            )

        super().__init__(name, **kwargs)

        if manifest is None:
            manifest = LazyManifest(())
        elif not isinstance(manifest, LazyManifest):
            manifest = LazyManifest(manifest)

        #: Immutable name-to-entry declarations.
        self.manifest: t.Final[LazyManifest] = manifest
        # Successfully loaded commands, keyed by manifest name. Only ever
        # mutated while holding ``_load_lock``. Use the read-only
        # :attr:`loaded_commands` view instead of touching this directly.
        self.commands: dict[str, Command] = {}
        self._loading: dict[str, _LoadState] = {}
        self._load_lock = threading.Lock()
        # Names currently being loaded by this thread, outermost first.
        self._loading_stack = threading.local()

    @property
    def loaded_commands(self) -> cabc.Mapping[str, Command]:
        """Read-only view of the commands loaded so far."""
        return MappingProxyType(self.commands)

    def add_command(self, cmd: Command, name: str | None = None) -> None:
        """Disabled: the manifest is immutable."""
        raise TypeError(
            "LazyGroup commands are declared in an immutable manifest and"
            " cannot be added at runtime."
        )

    def is_loaded(self, cmd_name: str) -> bool:
        """Return whether ``cmd_name`` has already been loaded."""
        with self._load_lock:
            return cmd_name in self.commands

    def _stack(self) -> list[str]:
        stack = getattr(self._loading_stack, "names", None)

        if stack is None:
            stack = []
            self._loading_stack.names = stack

        return stack

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:
        """Return the command for ``cmd_name``, loading it on demand.

        Returns ``None`` for names not in the manifest without calling
        any loader.
        """
        if cmd_name not in self.manifest:
            return None

        return self._load(cmd_name)

    def list_commands(self, ctx: Context) -> list[str]:
        """Manifest entry names in declaration order, without loading."""
        return list(self.manifest.ordered_names())

    def _load(self, cmd_name: str) -> Command:
        stack = self._stack()
        commands: dict[str, Command] = self.commands

        with self._load_lock:
            cached: Command | None = commands.get(cmd_name)

            if cached is not None:
                return cached

            # A loader that loops back to a name already being loaded
            # on this thread can never finish; report the whole chain.
            if cmd_name in stack:
                raise LazyCommandRecursionError([*stack, cmd_name])

            state = self._loading.get(cmd_name)

            if state is None:
                # This thread performs the load. The user loader runs
                # below, outside this lock.
                state = _LoadState()
                self._loading[cmd_name] = state
                stack.append(cmd_name)
                own_round = True
                entry = self.manifest[cmd_name]
            else:
                own_round = False
                entry = None

        if not own_round:
            # Another thread is loading. Block until that attempt
            # finishes and share its cached command or its failure.
            state.event.wait()

            with self._load_lock:
                cached = commands.get(cmd_name)

            if cached is not None:
                return cached

            assert state.error is not None
            raise state.error

        assert entry is not None

        try:
            command = entry.loader()
        except Exception as e:
            error = self._wrap_error(cmd_name, e)

            with self._load_lock:
                # The failed attempt is cleared so the next explicit
                # lookup retries with a fresh loader call.
                self._loading.pop(cmd_name, None)
                state.error = error
                state.event.set()

            # A wrapped error keeps the original as its cause; a
            # pre-built error (such as a recursion report) is raised
            # without an added cause.
            if error is e:
                raise
            raise error from e

        finally:
            stack.pop()

        if not isinstance(command, Command):
            error = LazyLoadError(
                cmd_name,
                TypeError(
                    "lazy loader must return a Command instance, got"
                    f" {type(command).__name__}"
                ),
            )

            with self._load_lock:
                self._loading.pop(cmd_name, None)
                state.error = error
                state.event.set()

            raise error

        with self._load_lock:
            # Exactly one command is ever published for the name.
            commands[cmd_name] = command
            self._loading.pop(cmd_name, None)
            state.event.set()

        return command

    @staticmethod
    def _wrap_error(cmd_name: str, error: Exception) -> BaseException:
        # Recursive-load reports already describe the chain; preserve them.
        if isinstance(error, LazyCommandRecursionError):
            return error

        if isinstance(error, ClickException):
            return error

        return LazyLoadError(cmd_name, error)

    def format_commands(self, ctx: Context, formatter: HelpFormatter) -> None:
        """List manifest entries using their short help; load nothing."""
        visible = [
            (name, entry.short_help)
            for name, entry in self.manifest.items()
            if not entry.hidden
        ]

        if not visible:
            return

        limit = formatter.width - 6 - max(len(name) for name, _ in visible)
        rows = [(name, _make_default_short_help(help, limit)) for name, help in visible]

        with formatter.section(_("Commands")):
            formatter.write_dl(rows)

    def shell_complete(self, ctx: Context, incomplete: str) -> list[CompletionItem]:
        """Complete manifest names from their short help; load nothing."""
        results = [
            CompletionItem(name, help=entry.short_help)
            for name, entry in self.manifest.items()
            if name.startswith(incomplete) and not entry.hidden
        ]
        results.extend(Command.shell_complete(self, ctx, incomplete))
        return results

    def resolve_command(
        self, ctx: Context, args: list[str]
    ) -> tuple[str | None, Command | None, list[str]]:
        # Same as Group.resolve_command, but suggestion candidates come
        # from the manifest instead of a mutable command map.
        from .exceptions import NoSuchCommand
        from .parser import _split_opt

        cmd_name = make_str(args[0])
        cmd = self.get_command(ctx, cmd_name)

        if cmd is None and ctx.token_normalize_func is not None:
            cmd_name = ctx.token_normalize_func(cmd_name)
            cmd = self.get_command(ctx, cmd_name)

        if cmd is None and not ctx.resilient_parsing:
            if _split_opt(cmd_name)[0]:
                self.parse_args(ctx, args)

            raise NoSuchCommand(
                cmd_name,
                possibilities=self.manifest.ordered_names(),
                ctx=ctx,
            )

        return cmd_name if cmd else None, cmd, args[1:]

    def to_info_dict(self, ctx: Context) -> dict[str, t.Any]:
        """Report manifest metadata without forcing every loader."""
        info_dict = Command.to_info_dict(self, ctx)
        with self._load_lock:
            loaded = set(self.commands)

        commands = {
            name: {
                "name": name,
                "short_help": entry.short_help,
                "hidden": entry.hidden,
                "loaded": name in loaded,
            }
            for name, entry in self.manifest.items()
        }
        info_dict.update(commands=commands, chain=self.chain)
        return info_dict
