"""Custom exception hierarchy for the SVM Cats-vs-Dogs pipeline.

Using a dedicated hierarchy (rather than bare Exception / ValueError)
lets calling code and logs distinguish failure domains at a glance and
enables precise except-clauses upstream (CLI, notebook, tests).
"""


class PipelineError(Exception):
    """Base class for all pipeline-specific errors."""


class ConfigError(PipelineError):
    """Raised when configuration loading or validation fails."""


class DataError(PipelineError):
    """Raised for dataset discovery, loading, or integrity problems."""


class FeatureExtractionError(PipelineError):
    """Raised when image feature extraction fails."""


class ModelError(PipelineError):
    """Raised for model training, persistence, or inference failures."""


class PredictionError(PipelineError):
    """Raised when inference on new/unseen data fails."""
