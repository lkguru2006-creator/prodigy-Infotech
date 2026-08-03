"""Project-specific exception hierarchy for precise error handling."""


class HousePricePredictorError(Exception):
    """Base exception for all pipeline errors."""


class DataValidationError(HousePricePredictorError):
    """Raised when input data fails schema or sanity checks."""


class DataIngestionError(HousePricePredictorError):
    """Raised when raw data cannot be loaded or generated."""


class FeatureEngineeringError(HousePricePredictorError):
    """Raised when feature transformations fail."""


class ModelTrainingError(HousePricePredictorError):
    """Raised when model fitting fails or produces invalid results."""


class ModelPersistenceError(HousePricePredictorError):
    """Raised when saving/loading model artifacts fails."""


class InferenceError(HousePricePredictorError):
    """Raised when prediction on new data fails."""
