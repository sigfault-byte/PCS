from typing import Any


def require_positive_int_id(value: Any, field_name: str) -> int:
    """Validate canonical 1-based integer ids loaded from JSON."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a positive int")

    if value < 1:
        raise ValueError(f"{field_name} must be >= 1")

    return value


def require_positive_int_ids(values: Any, field_name: str) -> list[int]:
    if values is None:
        return []

    if not isinstance(values, list):
        raise TypeError(f"{field_name} must be a list of positive ints")

    return [
        require_positive_int_id(value, f"{field_name}[{index}]")
        for index, value in enumerate(values)
    ]
