"""benchaudit -- screen a vision--language benchmark for protocol sensitivity."""
from .core import audit, AuditResult, RungResult, REG_INTERCEPT, REG_SLOPE
from .features import features, build_matrix, FEATURE_NAMES

__version__ = "0.1.0"
__all__ = ["audit", "AuditResult", "RungResult", "features", "build_matrix",
           "FEATURE_NAMES", "REG_INTERCEPT", "REG_SLOPE"]
