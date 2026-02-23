"""GPU and CPU memory release utilities for training runs."""

import ctypes
import gc
from collections.abc import Iterator
from typing import Any

import torch

_PARTIAL_ATTRS: tuple[str, ...] = ("for_training", "for_inference")


def _iter_model_chain(model: Any) -> Iterator[Any]:
    """Yield each node in the model.model.model... hierarchy chain.

    Args:
        model: The root model node.

    Yields:
        Each node in the chain until no further `.model` attribute exists.
    """
    node = model
    while node is not None:
        yield node
        node = getattr(node, "model", None)


def _break_trainer_refs(trainer: Any) -> None:
    """Break circular references held by a HuggingFace Trainer.

    Sets optimizer, lr_scheduler, and model references to None, clears
    callbacks, and calls accelerator.free_memory().  All operations share
    one exception handler so a partial failure is logged without blocking
    the remaining cleanup steps.

    Args:
        trainer: The HuggingFace Trainer instance, or None to skip.
    """
    if trainer is None:
        return
    try:
        trainer.optimizer = None
        trainer.lr_scheduler = None
        trainer.callback_handler.callbacks.clear()
        trainer.accelerator.free_memory()
        trainer.model = None
    except Exception as e:
        print(f"[cleanup] break_trainer_refs: {e}")


def _clear_fast_lora(model: Any) -> None:
    """Clear unsloth _fast_lora attributes from all model parameters.

    Args:
        model: The PEFT/unsloth model, or None to skip.
    """
    if model is None:
        return
    try:
        tuple(delattr(p, "_fast_lora") for p in model.parameters() if hasattr(p, "_fast_lora"))
    except Exception as e:
        print(f"[cleanup] clear_fast_lora: {e}")


def _clear_saved_temp_tokenizer(model: Any) -> None:
    """Clear _saved_temp_tokenizer from every node in the model hierarchy.

    Args:
        model: The PEFT/unsloth model, or None to skip.
    """
    if model is None:
        return
    try:
        tuple(
            delattr(node, "_saved_temp_tokenizer")
            for node in _iter_model_chain(model)
            if hasattr(node, "_saved_temp_tokenizer")
        )
    except Exception as e:
        print(f"[cleanup] clear_saved_temp_tokenizer: {e}")


def _clear_partial_refs(model: Any) -> None:
    """Clear functools.partial self-refs from every node in the model hierarchy.

    Removes for_training and for_inference attributes at each level of the
    model.model.model... chain.

    Args:
        model: The PEFT/unsloth model, or None to skip.
    """
    if model is None:
        return
    try:
        tuple(
            delattr(node, attr) for node in _iter_model_chain(model) for attr in _PARTIAL_ATTRS if hasattr(node, attr)
        )
    except Exception as e:
        print(f"[cleanup] clear_partial_refs: {e}")


def _load_libc() -> ctypes.CDLL | None:
    """Load libc for malloc_trim.

    Returns:
        The libc CDLL object, or None if unavailable on this platform.
    """
    try:
        return ctypes.CDLL("libc.so.6")
    except OSError:
        return None


def _flush_gpu_memory(libc: ctypes.CDLL | None) -> None:
    """Run one round of gc + CUDA cache flush + malloc_trim.

    Args:
        libc: The libc CDLL object, or None to skip malloc_trim.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if libc is not None:
        libc.malloc_trim(0)


def cleanup_training_run(
    trainer: Any = None,
    model: Any = None,
    tokenizer: Any = None,
    collator: Any = None,
    build_cache: Any = None,
    train_ds: Any = None,
    val_ds: Any = None,
    ds_all: Any = None,
    args: Any = None,
) -> None:
    """Release all GPU and CPU memory held by a training run.

    Breaks trainer internal circular references, clears unsloth-specific
    attributes, then deletes all objects in the correct order before
    flushing the CUDA allocator.

    Args:
        trainer: The HuggingFace Trainer instance.
        model: The PEFT/unsloth model.
        tokenizer: The tokenizer.
        collator: The data collator.
        build_cache: The text cache builder partial.
        train_ds: The training dataset.
        val_ds: The validation dataset.
        ds_all: The full dataset before splitting.
        args: The TrainingArguments instance.
    """
    _break_trainer_refs(trainer)
    _clear_fast_lora(model)
    _clear_saved_temp_tokenizer(model)
    _clear_partial_refs(model)

    del trainer, collator, build_cache, train_ds, val_ds, ds_all, args, model, tokenizer

    libc = _load_libc()
    tuple(_flush_gpu_memory(libc) for _ in range(3))
