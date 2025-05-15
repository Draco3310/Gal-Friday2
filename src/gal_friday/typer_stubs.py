"""Stub file for typer to satisfy imports in mypy checks.

This file provides minimal stub implementations for typer functions.
For actual functionality, the real typer package should be installed.
"""

from typing import Any, Callable, List, Optional, TypeVar

# Type variable for function arguments
T = TypeVar("T")


class Typer:
    """Stub class for Typer application."""

    def __init__(
        self,
        name: str = "",
        help: Optional[str] = None,
        no_args_is_help: bool = False,
        add_completion: bool = True,
    ) -> None:
        """Initialize a Typer application.

        Args
        ----
            name: The name of the CLI application
            help: A description for the CLI application
            no_args_is_help: Show help when no arguments are provided
            add_completion: Add completion for the application
        """
        self.name = name
        self.help = help
        self.no_args_is_help = no_args_is_help
        self.add_completion = add_completion
        self.commands: List[Callable] = []

    def command(
        self, name: Optional[str] = None, help: Optional[str] = None, **kwargs: Any
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Create a new command.

        Args
        ----
            name: Name for the command
            help: Help text to show for the command
            **kwargs: Additional options

        Returns
        -------
            Decorator function that registers the command
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self.commands.append(func)
            return func

        return decorator

    def run(self, **kwargs: Any) -> None:
        """Run the application.

        Args
        ----
            **kwargs: Additional arguments for running the application
        """


# Common typer functions stubs
def run(function: Callable[..., Any], **kwargs: Any) -> None:
    """Run a function as a Typer application.

    Args
    ----
        function: The function to run as a Typer application
        **kwargs: Additional arguments for running the application
    """


def Argument(
    default: Any = None,
    *,
    help: Optional[str] = None,
    show_default: bool = True,
    **kwargs: Any,
) -> Any:
    """Declare a command-line argument.

    Args
    ----
        default: Default value for the argument
        help: Help text for the argument
        show_default: Show the default value in help
        **kwargs: Additional options

    Returns
    -------
        An argument specification
    """
    return default


def Option(
    default: Any = None,
    *,
    help: Optional[str] = None,
    show_default: bool = True,
    is_flag: bool = False,
    **kwargs: Any,
) -> Any:
    """Create a command-line option.

    Args
    ----
        default: Default value for the option
        help: Help text for the option
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

    obj: Any

    def __init__(self, obj: Any = None) -> None:
        """Initialize a context.

        Args
        ----
            obj: Object to store in the context
        """
        self.obj = obj
