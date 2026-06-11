from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from p2predict.model_utils import extract_feature_info, inner_pipeline
from p2predict.trained_model_io import LoadModel


@dataclass
class ModelInfo:
    model_id: str
    path: str
    algorithm: str
    target: str
    features: list[str]
    feature_types: dict[str, str]
    categories: dict[str, list]
    log_target: bool
    r2: str
    training_date: str
    p2predict_version: str
    has_calibration: bool
    calibration_size: int | None
    has_background: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _model_info_from_loaded(model_id: str, path: str, loaded: dict) -> ModelInfo:
    """Build a ModelInfo from a loaded model dict."""
    pipeline = inner_pipeline(loaded["model"])
    feature_types, categories = extract_feature_info(pipeline)

    calibration = loaded.get("calibration")
    has_cal = bool(calibration and calibration.get("residuals"))
    cal_size = calibration.get("n_calibration") if has_cal else None

    return ModelInfo(
        model_id=model_id,
        path=path,
        algorithm=loaded.get("model_name", "unknown"),
        target=loaded.get("target_feature", "unknown"),
        features=list(loaded.get("features", [])),
        feature_types=feature_types,
        categories=categories,
        log_target=bool(loaded.get("log_target", False)),
        r2=str(loaded.get("r2", "")),
        training_date=str(loaded.get("training_date", "")),
        p2predict_version=str(loaded.get("p2predict_version", "")),
        has_calibration=has_cal,
        calibration_size=cal_size,
        has_background=loaded.get("background_sample") is not None,
    )


class ModelRegistry:
    """Scan, load, and cache .model files from a directory."""

    def __init__(self, models_dir: Path):
        self.models_dir = Path(models_dir)
        self._cache: dict[str, dict] = {}

    def _model_path(self, model_id: str) -> Path:
        return self.models_dir / f"{model_id}.model"

    def scan(self) -> list[ModelInfo]:
        """List all models in the directory with their metadata."""
        if not self.models_dir.exists():
            return []
        infos = []
        for path in sorted(self.models_dir.glob("*.model")):
            model_id = path.stem
            try:
                loaded = self.load(model_id)
                infos.append(_model_info_from_loaded(model_id, str(path), loaded))
            except Exception:
                continue
        return infos

    def load(self, model_id: str) -> dict:
        """Load a model by ID, using cache if available."""
        if model_id in self._cache:
            return self._cache[model_id]
        path = self._model_path(model_id)
        if not path.exists():
            raise FileNotFoundError(
                f"No model '{model_id}' found in {self.models_dir}"
            )
        loaded = LoadModel(str(path))
        self._cache[model_id] = loaded
        if len(self._cache) > 5:
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        return loaded

    def get_info(self, model_id: str) -> ModelInfo:
        """Get detailed metadata for a single model."""
        loaded = self.load(model_id)
        path = str(self._model_path(model_id))
        return _model_info_from_loaded(model_id, path, loaded)

    def register(self, model_id: str, path: Path, loaded: dict) -> ModelInfo:
        """Add a newly trained model to the cache."""
        self._cache[model_id] = loaded
        return _model_info_from_loaded(model_id, str(path), loaded)
