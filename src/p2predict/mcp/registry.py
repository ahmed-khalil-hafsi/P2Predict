from __future__ import annotations

from dataclasses import dataclass
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

    def _say_to_user(self) -> str:
        """One plain sentence to quote at discovery — no algorithm/R²/log-target.

        Deliberately makes NO trust claim: discovery has no holdout to judge on,
        so it points the agent at get_model_quality instead of implying a verdict.
        """
        n = len(self.features)
        specs = ", ".join(self.features) if self.features else "its specs"
        return (
            f"Estimates '{self.target}' from {n} spec(s) ({specs}). "
            "I haven't trust-checked this model yet — ask me to run a quality "
            "check before you benchmark a number against it."
        )

    def to_dict(self, include_internal: bool = False) -> dict:
        """Business-safe by default.

        A weaker agent surfaces whatever fields it sees, so the default view
        carries only what a category manager can hear: a plain `say_to_user`
        line, the target, the specs (and their types/categories, which the
        agent needs to build predict calls). The algorithm name, R², log-target
        flag, and calibration internals are gated behind ``include_internal``.
        """
        d = {
            "model_id": self.model_id,
            "say_to_user": self._say_to_user(),
            "target": self.target,
            "features": self.features,
            "feature_types": self.feature_types,
            "categories": self.categories,
            "training_date": self.training_date,
            "can_show_price_ranges": self.has_calibration,
        }
        if include_internal:
            d.update({
                "path": self.path,
                "algorithm": self.algorithm,
                "r2": self.r2,
                "log_target": self.log_target,
                "p2predict_version": self.p2predict_version,
                "calibration_size": self.calibration_size,
                "has_background": self.has_background,
            })
        return d


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
