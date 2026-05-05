from __future__ import annotations

from typing import Any, Iterable, TypeVar

T = TypeVar("T")

PROGRESS_BAR_FORMAT = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
PROGRESS_OPEN_FORMAT = "{desc}: {n_fmt} [{elapsed}, {rate_fmt}{postfix}]"


def progress_kwargs(**kwargs: Any) -> dict[str, Any]:
    kwargs.setdefault("bar_format", PROGRESS_OPEN_FORMAT if kwargs.get("total") is None else PROGRESS_BAR_FORMAT)
    kwargs.setdefault("dynamic_ncols", True)
    return kwargs


def progress_iter(iterable: Iterable[T], **kwargs: Any) -> Iterable[T]:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **progress_kwargs(**kwargs))


def progress_bar(**kwargs: Any) -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return None
    return tqdm(**progress_kwargs(**kwargs))
