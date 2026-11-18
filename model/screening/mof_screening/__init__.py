"""Material-specific adsorption-cycle screening.

Spatial and lumped representations share material laws, case validation,
execution guards and result accounting."""

from .contract import (
    FIXED_WEATHER,
    LUMPED,
    OFFICIAL_WEATHER,
    SPATIAL,
    MofScreeningError,
    build_case,
    load_defaults,
    preview_payload,
)
from .parallel_manager import ConcurrentMofScreeningManager as MofScreeningManager

__all__ = [
    "FIXED_WEATHER",
    "LUMPED",
    "MofScreeningError",
    "MofScreeningManager",
    "OFFICIAL_WEATHER",
    "SPATIAL",
    "build_case",
    "load_defaults",
    "preview_payload",
]
