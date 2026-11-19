"""A tiny name -> factory registry (used for the EJPS solvers).

::

    SOLVERS = Registry("ejps_solver")

    @SOLVERS.register("my_solver")
    class MySolver: ...

    solver = SOLVERS.build("my_solver", some_option=True)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any


class Registry:
    """Maps names to factories (classes or functions) of one component kind."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Any] = {}

    def register(self, name: str | None = None) -> Callable[[Any], Any]:
        """Decorator registering an object under ``name`` (default: its ``__name__``)."""

        def decorator(obj: Any) -> Any:
            key = name if name is not None else obj.__name__
            if not key:
                raise ValueError(f"{self.kind} registry: empty name")
            if key in self._items:
                raise KeyError(f"{self.kind} registry: {key!r} already registered")
            self._items[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"{self.kind} registry: unknown {name!r}; available: {self.names()}"
            ) from None

    def build(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Look up ``name`` and call it with the given arguments."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"
