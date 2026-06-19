from __future__ import annotations

import gc


def release_accelerator_memory() -> list[str]:
    """Best-effort cleanup between heavyweight model stages."""
    actions = ["gc.collect"]
    gc.collect()

    try:
        import torch
    except Exception:
        return actions

    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        actions.append("torch.cuda.empty_cache")
        try:
            torch.cuda.ipc_collect()
            actions.append("torch.cuda.ipc_collect")
        except Exception:
            pass

    mps = getattr(torch, "mps", None)
    if mps is not None and hasattr(mps, "empty_cache"):
        try:
            mps.empty_cache()
            actions.append("torch.mps.empty_cache")
        except Exception:
            pass

    return actions
