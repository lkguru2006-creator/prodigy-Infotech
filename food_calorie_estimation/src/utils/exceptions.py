"""Custom exception hierarchy for the food-calorie-estimation pipeline."""


class FoodCalorieError(Exception):
    """Base exception for all pipeline errors."""


class ConfigError(FoodCalorieError):
    """Raised when configuration is missing, malformed, or invalid."""


class DataGenerationError(FoodCalorieError):
    """Raised when synthetic data generation fails."""


class DataLoadError(FoodCalorieError):
    """Raised when raw/processed data cannot be loaded."""


class FeatureExtractionError(FoodCalorieError):
    """Raised when feature extraction (fit/transform) fails."""


class ModelError(FoodCalorieError):
    """Raised for model training, prediction, save/load failures."""


class CalorieEstimationError(FoodCalorieError):
    """Raised when calorie lookup/estimation fails."""
