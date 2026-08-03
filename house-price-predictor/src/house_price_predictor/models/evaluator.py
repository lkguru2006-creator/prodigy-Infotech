"""
Model evaluation.

Computes a standard regression metric suite (RMSE, MAE, R2, MAPE) and
returns a structured, serializable result rather than printing to stdout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from house_price_predictor.utils.exceptions import ModelTrainingError
from house_price_predictor.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    rmse: float
    mae: float
    r2: float
    mape: float
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)


class ModelEvaluator:
    """Computes regression metrics comparing predictions against ground truth."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> EvaluationResult:
        try:
            y_true = np.asarray(y_true, dtype=float)
            y_pred = np.asarray(y_pred, dtype=float)

            if y_true.shape[0] == 0:
                raise ModelTrainingError("Cannot evaluate on empty arrays.")
            if y_true.shape != y_pred.shape:
                raise ModelTrainingError(
                    f"Shape mismatch in evaluation: y_true={y_true.shape}, "
                    f"y_pred={y_pred.shape}"
                )

            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mae = float(mean_absolute_error(y_true, y_pred))
            r2 = float(r2_score(y_true, y_pred))
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

            result = EvaluationResult(
                rmse=rmse, mae=mae, r2=r2, mape=mape, n_samples=int(y_true.shape[0])
            )
            logger.info(
                "Evaluation -> RMSE: %.2f | MAE: %.2f | R2: %.4f | MAPE: %.2f%% | n=%d",
                result.rmse,
                result.mae,
                result.r2,
                result.mape,
                result.n_samples,
            )
            return result

        except ModelTrainingError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelTrainingError(f"Evaluation failed: {exc}") from exc
