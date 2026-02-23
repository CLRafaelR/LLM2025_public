"""Pytest configuration and shared fixtures for GPU memory management."""

import ctypes
import gc
from collections.abc import Generator

import pytest
import torch

# import wandb


def pytest_configure() -> None:
    """Set CUDA allocator config before any CUDA initialization.

    Must run before collection/imports trigger CUDA init.  The ``config``
    parameter is intentionally omitted because it is not needed here;
    pytest only passes arguments that the hook function declares.
    """
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


@pytest.fixture(autouse=True)
def gpu_cleanup_guard() -> Generator[None, None, None]:
    """Ensure GPU memory is released between test runs.

    Logs allocated/reserved MB before and after cleanup.
    Runs 3 rounds of gc + empty_cache + synchronize + malloc_trim.

    Yields:
        None
    """
    yield

    if not torch.cuda.is_available():
        return

    allocated_before = torch.cuda.memory_allocated() / 1024**2
    reserved_before = torch.cuda.memory_reserved() / 1024**2
    print(f"\n[GPU] Before cleanup: allocated={allocated_before:.1f}MB, reserved={reserved_before:.1f}MB")

    try:
        libc = ctypes.CDLL("libc.so.6")
    except OSError:
        libc = None

    def _one_round() -> None:
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if libc is not None:
            libc.malloc_trim(0)

    tuple(_one_round() for _ in range(3))

    allocated_after = torch.cuda.memory_allocated() / 1024**2
    reserved_after = torch.cuda.memory_reserved() / 1024**2
    print(f"[GPU] After cleanup:  allocated={allocated_after:.1f}MB, reserved={reserved_after:.1f}MB")


# @pytest.fixture(autouse=True)
# def wandb_cleanup_guard() -> Generator[None, None, None]:
#     """Finish any active wandb run after each test.
#
#     Yields:
#         None
#     """
#     yield
#
#     try:
#         wandb.finish(quiet=True)
#     except Exception:
#         pass
