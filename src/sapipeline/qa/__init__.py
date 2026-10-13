"""Read-only technical image, annotation and ROI quality gates."""

from .validator import QAResult, validate_artifact

__all__ = ["QAResult", "validate_artifact"]
