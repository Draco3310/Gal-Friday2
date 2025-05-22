"""Stub file for typer to satisfy imports in mypy checks.

This file provides minimal stub implementations for typer functions.
For actual functionality, the real typer package should be installed.
"""

from typing import Optional, TypeVar

from collections.abc import Callable

# Type variable for function arguments
T = TypeVar("T")


class Typer:
    """Stub class for Typer application."""

    def __init__(
        self,
        name: str = "",
        help_text: str | None = None,
        no_args_is_help: bool = False,
        add_completion: bool = True,
    ) -> None:
        """Initialize a Typer application.

        Args
        ----
            name: The name of the CLI application
            help_text: A description for the CLI application
            no_args_is_help: Show help when no arguments are provided
            add_completion: Add completion for the application
        """
        self.name = name
        self.help_text = help_text
        self.no_args_is_help = no_args_is_help
        self.add_completion = add_completion
        self.commands: list[Callable] = []

    def command(
        self, _name: str | None = None, _help_text: str | None = None, **_kwargs: object
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Create a new command.

        Args
        ----
            name: Name for the command
            help_text: Help text to show for the command
            **kwargs: Additional options

        Returns
        -------
            Decorator function that registers the command
        """

        def decorator(func: Callable[..., object]) -> Callable[..., object]:
            self.commands.append(func)
            return func

        return decorator

    def run(self, **_kwargs: object) -> None:
        """Run the application.

        Args
        ----
            **kwargs: Additional arguments for running the application
        """


# Common typer functions stubs
def run(function: Callable[..., object], **_kwargs: object) -> None:
    """Run a function as a Typer application.

    Args
    ----
        function: The function to run as a Typer application
        **kwargs: Additional arguments for running the application
    """


def argument(
    default: object = None,
    *,
    _help_text: str | None = None,
    _show_default: bool = True,
    **_kwargs: object,
) -> object:
    """Declare a command-line argument.

    Args
    ----
        default: Default value for the argument
        help_text: Help text for the argument
        show_default: Show the default value in help
        **kwargs: Additional options

    Returns
    -------
        An argument specification
    """
    return default


def option(
    default: object = None,
    *,
    _help_text: str | None = None,
    _show_default: bool = True,
    _is_flag: bool = False,
    **_kwargs: object,
) -> object:
    """Create a command-line option.

    Args
    ----
        default: Default value for the option
        help_text: Help text for the option
        show_default: Show the default value in help
        is_flag: Whether the option is a flag (boolean)
        **kwargs: Additional options

    Returns
    -------
        An option specification
    """
    return default


class Context:
    """A context object for typer commands."""

    obj: object

    def __init__(self, obj: object = None) -> None:
        """Initialize a context.

        Args
        ----
            obj: Object to store in the context
        """
        self.obj = obj
