import functools
import os
import pickle
from typing import get_type_hints


def save_result(obj, filepath, save_fn=None):
    if save_fn:
        return save_fn(obj, filepath)
    if hasattr(obj, "to_pickle"):
        return obj.to_pickle(filepath)
    if hasattr(obj, "save"):
        return obj.save(filepath)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_result(filepath, load_fn=None, expected_type=None):
    if load_fn:
        return load_fn(filepath)
    if expected_type and hasattr(expected_type, "load"):
        return expected_type.load(filepath)
    if expected_type and hasattr(expected_type, "from_pickle"):
        return expected_type.from_pickle(filepath)
    with open(filepath, "rb") as f:
        return pickle.load(f)


def with_file_cache(
        default_filepath=None,
        default_mode="calculate",
        expected_type=None,
        save_fnc=None, load_fnc=None
        ):
    """
    Decorator to cache function results to/from a file.
    kwargs:
      - filepath: str (required for load/save)
      - mode: {"calculate","save","load","auto"}
      - save_fn / load_fn: optional callables to override persistence
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mode = kwargs.pop("mode", default_mode)
            filepath = kwargs.pop("filepath", default_filepath)
            actual_expected_type = expected_type or get_type_hints(func).get("return")

            # 1) Strict load mode: must have a path and file must exist
            if mode == "load":
                if not filepath:
                    raise ValueError("`filepath` is required in load mode.")
                if not os.path.exists(filepath):
                    raise FileNotFoundError(f"No saved result at {filepath}")
                return load_result(filepath, load_fnc, actual_expected_type)

            # 2) Auto mode: load if available, otherwise compute and (later) save
            if mode == "auto" and filepath and os.path.exists(filepath):
                return load_result(filepath, load_fnc, actual_expected_type)

            # 3) Compute
            result = func(*args, **kwargs)

            # 4) Save if requested
            if mode in ("save", "auto"):
                if not filepath and mode == "save":
                    raise ValueError("`filepath` is required in save mode.")
                save_result(result, filepath, save_fnc)

            return result
        return wrapper
    return decorator
