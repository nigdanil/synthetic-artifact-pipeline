"""Pure deterministic selection of bbox *instances*, not segmentation masks."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from sapipeline.annotations import AnnotationDocument, ObjectAnnotation

SERVICE_CLASSES = frozenset({"glint", "obstacle_object", "other_shelf", "other_lary"})
GENERIC_CLASS = "product"  # Optional random target, but selectable explicitly.
SelectionMode = Literal["by_index", "by_class", "random_one", "random_n"]


class SelectionError(ValueError):
    """Invalid selection or no eligible target: never silently pick background."""


@dataclass(frozen=True, slots=True)
class SelectionResult:
    mode: SelectionMode
    seed: int
    targets: tuple[ObjectAnnotation, ...]


def select_rois(
    document: AnnotationDocument,
    *,
    mode: SelectionMode,
    seed: int = 0,
    instance_index: int | None = None,
    class_name: str | None = None,
    count: int = 1,
    include_generic_product: bool = False,
    allow_service_targets: bool = False,
) -> SelectionResult:
    """Choose unique instances by index/class or reproducibly from merchandise boxes.

    Random modes never include service classes, regardless of overrides. A service
    class may be selected only by explicit index/name *and* allow_service_targets.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SelectionError("seed must be a non-negative integer")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise SelectionError("count must be a positive integer")
    if mode not in ("by_index", "by_class", "random_one", "random_n"):
        raise SelectionError(f"Unsupported selection mode: {mode!r}")
    objects = document.objects
    if mode == "by_index":
        if isinstance(instance_index, bool) or not isinstance(instance_index, int) or instance_index < 0:
            raise SelectionError("by_index requires a non-negative instance_index")
        targets = tuple(obj for obj in objects if obj.instance_index == instance_index)
        if len(targets) != 1:
            raise SelectionError(f"No unique instance_index={instance_index}")
    elif mode == "by_class":
        if not isinstance(class_name, str) or not class_name:
            raise SelectionError("by_class requires an exact, non-empty class_name")
        targets = tuple(obj for obj in objects if obj.class_name == class_name)
        if not targets:
            raise SelectionError(f"No objects with class_name={class_name!r}")
    else:
        eligible = [obj for obj in objects if obj.class_name not in SERVICE_CLASSES
                    and (include_generic_product or obj.class_name != GENERIC_CLASS)]
        n = count if mode == "random_n" else 1
        if n > len(eligible):
            raise SelectionError(f"Requested {n} product ROI(s), only {len(eligible)} eligible")
        chosen = random.Random(seed).sample(eligible, n)
        targets = tuple(sorted(chosen, key=lambda obj: obj.instance_index))
    if not allow_service_targets and any(obj.class_name in SERVICE_CLASSES for obj in targets):
        raise SelectionError("Service ROI requires explicit --allow-service-targets")
    if len({obj.instance_index for obj in targets}) != len(targets):
        raise SelectionError("Duplicate instance_index in selected targets")
    return SelectionResult(mode=mode, seed=seed, targets=targets)
