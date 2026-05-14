"""
InferenceEngine — the single public façade over the six-phase stack.

Responsibilities:
    - Load & hold one ``FeaturePipeline`` + one ``NeuralThompson``.
    - Lazily build an ``ExplainabilityExtractor`` (+ optional
      ``AttributionEngine``) and an ``LLMExplainer`` (+ client).
    - Validate every caller input through Pydantic schemas.
    - Serve single/batch predictions (sync + async).
    - Serve continuous-learning updates (single row, iterable, CSV, or
      session-scoped) under a single write lock.
    - Emit drift alerts via an embedded ``DriftMonitor``.

Nothing else. Training, persistence of logs, champion/challenger, and
feature-engineering changes live elsewhere (see design §9).
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import threading
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import ValidationError as PydanticValidationError

from ._internal.constants import CONTEXT_FEATURES, TREATMENTS
from ._internal.explainability import ExplainabilityExtractor, SubgroupRegret
from ._internal.feature_engineering import ALL_FEATURES, FeaturePipeline
from ._internal.interpretability import AttributionEngine
from ._internal.monitoring import DriftMonitor
from ._internal.neural_bandit import NeuralThompson

from .config import InferenceConfig
from .errors import (
    ConfigurationError,
    ExplanationError,
    ModelError,
    ValidationError,
)
from .schemas import LearningAck, LearningRecord, PatientInput, PredictionResult
from .stub_client import StubClient


ExplainArg = Union[bool, Literal["require"]]


class InferenceEngine:
    """
    Single-tenant inference engine. Thread-safe for concurrent reads; writes
    (updates) are serialised via an ``RLock``. Predictions do **not** acquire
    the write lock — the worst-case view is one update stale, acceptable by
    design (see module docstring / design §6.4).
    """

    # ── construction ──────────────────────────────────────────────────────

    def __init__(
        self,
        config: InferenceConfig,
        pipeline: FeaturePipeline,
        model: NeuralThompson,
        *,
        extractor: Optional[ExplainabilityExtractor] = None,
        llm_explainer: Any = None,
        drift_monitor: Optional[DriftMonitor] = None,
    ):
        self.config = config
        self.pipeline = pipeline
        self.model = model
        self._extractor = extractor
        self._llm_explainer = llm_explainer
        self._drift = drift_monitor

        self._lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None
        self._n_updates = 0
        self._ready = True

        self._pipeline_version = _hash_path(config.pipeline_path)
        self._model_version = _hash_path(config.model_path)

        self._subgroup_regrets: Optional[List[SubgroupRegret]] = None

    @classmethod
    def from_config(cls, config: InferenceConfig) -> "InferenceEngine":
        pipeline = _load_pipeline(config.pipeline_path)
        _validate_pipeline_schema(pipeline)
        model = _load_model(config)

        if config.online_retraining:
            model.enable_online_retraining(
                buffer_size=config.replay_buffer_size,
                retrain_every=config.retrain_every,
                minibatch_size=config.minibatch_size,
                retrain_epochs=config.retrain_epochs,
                min_buffer_for_retrain=config.min_buffer_for_retrain,
            )

        attribution_engine: Optional[AttributionEngine] = None
        if config.attribution_enabled:
            attribution_engine = AttributionEngine(
                feature_names=list(pipeline.features),
            )

        extractor = ExplainabilityExtractor(
            model=model,
            n_confidence_draws=config.n_confidence_draws,
            attribution_engine=attribution_engine,
        )

        drift = None
        if config.drift_enabled:
            drift = DriftMonitor(
                baseline_size=config.drift_baseline_size,
                window_size=config.drift_window_size,
                threshold_z=config.drift_threshold_z,
            )

        return cls(
            config=config,
            pipeline=pipeline,
            model=model,
            extractor=extractor,
            drift_monitor=drift,
        )

    @classmethod
    def from_env(cls) -> "InferenceEngine":
        return cls.from_config(InferenceConfig.load())

    # ── introspection ─────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready and self.pipeline is not None and self.model is not None

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ready": self.ready,
            "n_updates": self._n_updates,
            "model_path": str(self.config.model_path),
            "pipeline_path": str(self.config.pipeline_path),
            "model_version": self._model_version,
            "pipeline_version": self._pipeline_version,
            "feature_names": list(self.pipeline.features),
            "llm_enabled": self.config.llm_enabled,
            "llm_provider": self.config.llm_provider,
            "online_retraining": self.config.online_retraining,
            "drift_enabled": self.config.drift_enabled,
            "n_confidence_draws": self.config.n_confidence_draws,
            "attribution_enabled": self.config.attribution_enabled,
        }

    # ── prediction (sync) ─────────────────────────────────────────────────

    def predict(
        self,
        patient: Union[Dict[str, Any], PatientInput],
        explain: ExplainArg = False,
    ) -> PredictionResult:
        pi = self._validate_patient(patient, context="patient")
        x = self._transform(pi)
        try:
            payload = self._extractor.extract(pi.context_dict(), x)
        except Exception as e:
            raise ModelError(f"Model forward pass failed: {e}") from e

        explanation = self._maybe_explain(payload, explain)

        if self._drift is not None:
            self._drift.observe(context_norm=float(np.linalg.norm(x)))

        return PredictionResult.from_payload(
            payload,
            patient_id=pi.patient_id,
            explanation=explanation,
            model_version=self._model_version,
            pipeline_version=self._pipeline_version,
        )

    def predict_batch(
        self,
        patients: Union[pd.DataFrame, Iterable[Dict[str, Any]]],
        explain: ExplainArg = False,
    ) -> List[PredictionResult]:
        rows = _to_records(patients)
        results: List[PredictionResult] = []
        for raw in rows:
            try:
                pi = PatientInput.model_validate(raw)
            except PydanticValidationError as e:
                results.append(PredictionResult.rejected(
                    errors=e.errors(),
                    patient_id=raw.get("patient_id") if isinstance(raw, dict) else None,
                ))
                continue
            try:
                results.append(self.predict(pi, explain=explain))
            except ValidationError as e:
                results.append(PredictionResult.rejected(
                    errors=e.errors(), patient_id=pi.patient_id,
                ))
            except (ModelError, ExplanationError) as e:
                if isinstance(e, ExplanationError) and explain != "require":
                    # should not reach here — predict already soft-catches
                    logger.warning(f"Unexpected ExplanationError in batch: {e}")
                results.append(PredictionResult.rejected(
                    errors=[{"loc": [], "msg": str(e), "type": type(e).__name__}],
                    patient_id=pi.patient_id,
                ))
        return results

    # ── prediction (async) ────────────────────────────────────────────────

    async def apredict(
        self,
        patient: Union[Dict[str, Any], PatientInput],
        explain: ExplainArg = False,
    ) -> PredictionResult:
        return await asyncio.to_thread(self.predict, patient, explain)

    # ── continuous learning (sync) ────────────────────────────────────────

    def update(self, record: Union[Dict[str, Any], LearningRecord]) -> LearningAck:
        try:
            rec = record if isinstance(record, LearningRecord) \
                else LearningRecord.model_validate(record)
        except PydanticValidationError as e:
            return LearningAck(
                accepted=False,
                n_updates_so_far=self._n_updates,
                validation_errors=e.errors(),
            )

        with self._lock:
            try:
                x = self._transform(rec.patient)
                before_step = getattr(self.model, "_online_step", 0)
                self.model.online_update(x, int(rec.action), float(rec.reward))
                after_step = getattr(self.model, "_online_step", 0)
                retrained = bool(
                    self.config.online_retraining
                    and self.config.retrain_every > 0
                    and after_step > 0
                    and after_step != before_step
                    and after_step % self.config.retrain_every == 0
                )
                self._n_updates += 1

                alerts: List[Dict[str, Any]] = []
                if self._drift is not None:
                    drift_alerts = self._drift.observe(
                        context_norm=float(np.linalg.norm(x)),
                        action=int(rec.action),
                        reward=float(rec.reward),
                    )
                    alerts = [a.to_dict() for a in drift_alerts]

                return LearningAck(
                    accepted=True,
                    n_updates_so_far=self._n_updates,
                    posterior_updated=True,
                    backbone_retrained=retrained,
                    drift_alerts=alerts,
                    patient_id=rec.patient.patient_id,
                    observed_at=rec.observed_at,
                )
            except Exception as e:
                logger.warning(f"update() failed: {e}")
                return LearningAck(
                    accepted=False,
                    n_updates_so_far=self._n_updates,
                    validation_errors=[{
                        "loc": [], "msg": str(e), "type": type(e).__name__,
                    }],
                    patient_id=getattr(rec.patient, "patient_id", None),
                )

    def update_many(
        self,
        records: Iterable[Union[Dict[str, Any], LearningRecord]],
    ) -> Iterator[LearningAck]:
        for r in records:
            yield self.update(r)

    def ingest_csv(
        self,
        path: Union[str, Path],
        *,
        encoding: str = "utf-8",
    ) -> Iterator[LearningAck]:
        p = Path(path)
        if not p.exists():
            raise ConfigurationError(f"CSV not found: {p}")
        with p.open("r", encoding=encoding, newline="") as fh:
            reader = csv.DictReader(fh)
            required = {"reward"}
            action_or_treatment = {"action", "treatment"}
            headers = set(reader.fieldnames or [])
            if not required.issubset(headers):
                raise ValidationError(
                    f"CSV missing required columns: {required - headers}"
                )
            if not (headers & action_or_treatment):
                raise ValidationError(
                    "CSV must include either 'action' or 'treatment' column"
                )

            def _gen() -> Iterator[Dict[str, Any]]:
                for row in reader:
                    yield _coerce_csv_row(row)

            yield from self.update_many(_gen())

    # ── continuous learning (async) ───────────────────────────────────────

    async def aupdate(
        self,
        record: Union[Dict[str, Any], LearningRecord],
    ) -> LearningAck:
        return await asyncio.to_thread(self.update, record)

    async def aupdate_many(
        self,
        records: Union[
            Iterable[Union[Dict[str, Any], LearningRecord]],
            AsyncIterable[Union[Dict[str, Any], LearningRecord]],
        ],
    ) -> AsyncIterator[LearningAck]:
        if hasattr(records, "__aiter__"):
            async for r in records:  # type: ignore[union-attr]
                yield await self.aupdate(r)
        else:
            for r in records:  # type: ignore[union-attr]
                yield await self.aupdate(r)

    # ── sessions ──────────────────────────────────────────────────────────

    @contextmanager
    def learning_session(
        self,
        *,
        checkpoint_every: Optional[int] = None,
        emit_metrics: bool = True,
    ):
        from .streaming import LearningSession
        session = LearningSession(
            self,
            checkpoint_every=checkpoint_every,
            emit_metrics=emit_metrics,
        )
        try:
            yield session
        finally:
            session.close()

    @asynccontextmanager
    async def alearning_session(
        self,
        *,
        checkpoint_every: Optional[int] = None,
        emit_metrics: bool = True,
    ):
        from .streaming import AsyncLearningSession
        session = AsyncLearningSession(
            self,
            checkpoint_every=checkpoint_every,
            emit_metrics=emit_metrics,
        )
        try:
            yield session
        finally:
            await session.aclose()

    # ── rich event stream ────────────────────────────────────────────────
    @contextmanager
    def learning_stream(
        self,
        *,
        total_steps: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """
        Context-managed Thompson-sampling stream that emits
        :class:`LearningStepEvent` per step. Does not replace
        ``learning_session`` — it is a separate, richer channel for
        real-time observability.
        """
        from .events import LearningStream
        stream = LearningStream(self, total_steps=total_steps, rng=rng)
        try:
            yield stream
        finally:
            stream.close()

    @asynccontextmanager
    async def alearning_stream(
        self,
        *,
        total_steps: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        """Async mirror of :meth:`learning_stream`."""
        from .events import LearningStream
        stream = LearningStream(self, total_steps=total_steps, rng=rng)
        try:
            yield stream
        finally:
            stream.close()

    # ── checkpointing ─────────────────────────────────────────────────────

    def checkpoint(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Persist the current NeuralThompson (backbone + posterior) to disk."""
        target = Path(path) if path is not None \
            else (self.config.checkpoint_dir or self.config.model_path.parent) \
            / f"neural_thompson_snapshot_{self._n_updates}.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target))
        logger.info(f"InferenceEngine checkpoint written: {target}")
        return target

    # ── internals ─────────────────────────────────────────────────────────

    def _validate_patient(
        self, patient: Union[Dict[str, Any], PatientInput], *, context: str,
    ) -> PatientInput:
        if isinstance(patient, PatientInput):
            return patient
        try:
            return PatientInput.model_validate(patient)
        except PydanticValidationError as e:
            raise ValidationError(
                f"Invalid {context} input", errors=e.errors(), source=patient,
            ) from e

    def _transform(self, pi: PatientInput) -> np.ndarray:
        try:
            x = self.pipeline.transform_single(pi.feature_dict())
        except Exception as e:
            raise ModelError(f"Feature pipeline transform failed: {e}") from e
        if not np.all(np.isfinite(x)):
            raise ModelError("Feature pipeline produced non-finite values")
        return x

    def _maybe_explain(
        self, payload: Dict[str, Any], explain: ExplainArg,
    ) -> Optional[Dict[str, Any]]:
        if not explain:
            return None
        require = explain == "require"
        try:
            explainer = self._get_llm_explainer()
        except ConfigurationError:
            if require:
                raise
            logger.warning("LLM explainer unavailable; returning explanation=None")
            return None
        try:
            return explainer.explain(payload)
        except Exception as e:
            if require:
                raise ExplanationError(str(e)) from e
            logger.warning(f"LLM explanation failed (soft): {e}")
            return None

    def _get_llm_explainer(self):
        if self._llm_explainer is not None:
            return self._llm_explainer
        cfg = self.config
        if not cfg.llm_enabled:
            raise ConfigurationError(
                "explain=True but config.llm_enabled is False — "
                "set BANDITS_LLM_ENABLED=true or pass llm_enabled=True"
            )
        from ._internal.llm_explain import LLMExplainer  # lazy import

        if cfg.llm_provider == "stub":
            client = StubClient()
        elif cfg.llm_provider == "gemini":
            from ._internal.llm_explain import GeminiClient
            key = cfg.resolve_api_key()
            if not key:
                raise ConfigurationError(
                    "Gemini provider selected but no API key found "
                    "(set BANDITS_LLM_API_KEY or GEMINI_API_KEY)"
                )
            client = GeminiClient(
                api_key=key,
                model_name=cfg.llm_model_name,
                temperature=cfg.llm_temperature,
            )
        else:
            raise ConfigurationError(
                f"Unknown llm_provider={cfg.llm_provider!r}"
            )
        self._llm_explainer = LLMExplainer(
            max_retries=cfg.llm_max_retries,
            client=client,
        )
        return self._llm_explainer


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline(path: Path) -> FeaturePipeline:
    if not Path(path).exists():
        raise ConfigurationError(f"Feature pipeline not found at {path}")
    try:
        return FeaturePipeline.load(str(path))
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load feature pipeline from {path}: {e}"
        ) from e


def _load_model(config: InferenceConfig) -> NeuralThompson:
    path = Path(config.model_path)
    if not path.exists():
        raise ConfigurationError(f"Model checkpoint not found at {path}")
    import torch

    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    input_dim = cfg.get("input_dim")
    if input_dim is None:
        raise ConfigurationError(
            f"Model checkpoint at {path} missing input_dim in config block"
        )
    device = config.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralThompson(
        input_dim=int(input_dim),
        hidden_dims=cfg.get("hidden_dims"),
        dropout=cfg.get("dropout", 0.1),
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-4),
        reg_lambda=cfg.get("reg_lambda", 1.0),
        noise_variance=cfg.get("noise_variance", 0.25),
        device=device,
    )
    try:
        model.load(str(path))
    except Exception as e:
        raise ConfigurationError(
            f"Failed to load NeuralThompson weights from {path}: {e}"
        ) from e
    return model


def _validate_pipeline_schema(pipeline: FeaturePipeline) -> None:
    expected = set(ALL_FEATURES)
    got = set(pipeline.features)
    if got != expected:
        diff = (expected ^ got)
        raise ConfigurationError(
            f"Feature pipeline feature list does not match expected schema "
            f"(differences: {sorted(diff)}). Retrain with the current "
            f"feature_engineering.ALL_FEATURES or update the schema."
        )


def _hash_path(path: Path) -> str:
    p = Path(path)
    if not p.exists():
        return "unknown"
    h = hashlib.sha1()
    h.update(str(p).encode("utf-8"))
    h.update(str(p.stat().st_mtime_ns).encode("utf-8"))
    h.update(str(p.stat().st_size).encode("utf-8"))
    return h.hexdigest()[:12]


def _to_records(
    patients: Union[pd.DataFrame, Iterable[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    if isinstance(patients, pd.DataFrame):
        return patients.to_dict(orient="records")
    return list(patients)


def _coerce_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
    """Turn a CSV-reader string row into a LearningRecord-compatible dict."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if v is None or v == "":
            continue
        if k == "treatment":
            out[k] = v
            continue
        if k in {"action", "reward", "age", "bmi", "hba1c_baseline", "egfr",
                 "diabetes_duration", "fasting_glucose", "c_peptide",
                 "bp_systolic", "ldl", "hdl", "triglycerides", "alt",
                 "cvd", "ckd", "nafld", "hypertension",
                 "medullary_thyroid_history", "men2_history",
                 "pancreatitis_history", "type1_suspicion"}:
            try:
                f = float(v)
                out[k] = int(f) if f.is_integer() and k != "reward" else f
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


__all__ = ["InferenceEngine", "ExplainArg"]
