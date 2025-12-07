"""CLI shim for backward compatibility."""

from ics.predict.predictor import (
    IncidentPredictor,
    PredictionResult,
    DEFAULT_MULTIPLIERS,
    main,
)

__all__ = ["IncidentPredictor", "PredictionResult", "DEFAULT_MULTIPLIERS", "main"]

if __name__ == "__main__":
    main()
