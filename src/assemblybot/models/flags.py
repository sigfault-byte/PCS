from enum import IntFlag, auto


class SegmentFlag(IntFlag):
    NONE = 0
    IS_SHORT = auto()
    HAS_OVERLAP = auto()
    MULTIPLE_SPEAKERS = auto()
    NON_FRENCH = auto()
    GIBBERISH = auto()
    NEEDS_REVIEW = auto()


def flags_to_list(flags: int) -> list[str | None]:
    return [f.name for f in SegmentFlag if f & flags]
