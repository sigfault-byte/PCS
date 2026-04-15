from enum import IntFlag, auto


class SegmentFlag(IntFlag):
    NONE = 0
    IS_SHORT = auto()  # 2^0 = 1
    HAS_OVERLAP = auto()  # 2^1 = 2
    MULTIPLE_SPEAKERS = auto()  # 2^2 = 4
    NON_FRENCH = auto()  # 2^3 = 8
    GIBBERISH = auto()  # 2^4 = 16
    NEEDS_REVIEW = auto()  # 2^5 = 32


def flags_to_list(flags: int) -> list[str | None]:
    return [f.name for f in SegmentFlag if f & flags]
