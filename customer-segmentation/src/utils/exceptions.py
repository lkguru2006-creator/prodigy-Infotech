"""
Custom exception hierarchy for the customer segmentation pipeline.

Using typed exceptions (rather than bare Exception/ValueError) lets callers
catch precisely what they expect and lets the pipeline produce clear,
actionable error messages instead of opaque stack traces.
"""


class CustomerSegmentationError(Exception):
    """Base exception for all pipeline-specific errors."""


class ConfigurationError(CustomerSegmentationError):
    """Raised when configuration is missing, malformed, or invalid."""


class DataValidationError(CustomerSegmentationError):
    """Raised when input data fails schema or quality validation."""


class DataLoadError(CustomerSegmentationError):
    """Raised when raw data cannot be located or read."""


class FeatureEngineeringError(CustomerSegmentationError):
    """Raised when feature transformation fails."""


class ModelTrainingError(CustomerSegmentationError):
    """Raised when model fitting fails."""


class ModelPersistenceError(CustomerSegmentationError):
    """Raised when saving or loading model artifacts fails."""


class EvaluationError(CustomerSegmentationError):
    """Raised when cluster evaluation/metrics computation fails."""
