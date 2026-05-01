from __future__ import annotations

from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def progress_iter(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, dynamic_ncols=True, **kwargs)


def progress_bar(**kwargs: Any) -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(dynamic_ncols=True, **kwargs)
