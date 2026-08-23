"""Trait-related helpers shared by widget constructors."""

from __future__ import annotations


def reject_unknown_kwargs(cls, kwargs: dict) -> None:
    """Raise TypeError for any kwarg that isn't a declared trait (catches typos)."""
    traits = set(cls.class_trait_names())
    unknown = [k for k in kwargs if k not in traits]
    if unknown:
        key = sorted(unknown)[0]
        raise TypeError(f"{cls.__name__}() got unexpected keyword argument {key!r}.")
