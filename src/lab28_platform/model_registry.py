"""MLflow as the release registry for the RAG service (IP06).

What gets registered here is a *release*, not a set of weights. The lab serves
one pinned model through vLLM, so the thing that actually changes between
deployments — and the thing an incident review needs to identify — is the
combination of prompt, retrieval configuration, served model id, and the data
version the release was evaluated against. That bundle is what earns a version
number and the ``champion`` alias.

Aliases rather than stages: a stage name says where a version is meant to be,
an alias says which version is actually serving. Rollback is then one call that
moves the alias back, and the serving path notices on its next refresh.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from lab28_platform import metrics
from lab28_platform.settings import MLflowSettings
from lab28_platform.telemetry import SPAN_MLFLOW_RESOLVE, span

#: Tags every version carries so a version alone answers "made from what?".
TAG_PROMPT_VERSION = "lab28.prompt_version"
TAG_VLLM_MODEL = "lab28.vllm_model_id"
TAG_EMBEDDING_MODEL = "lab28.embedding_model_id"
TAG_DELTA_VERSION = "lab28.delta_version"
TAG_COLLECTION = "lab28.qdrant_collection"
TAG_FEATURE_SERVICE = "lab28.feature_service"


@contextmanager
def _portable_mlflow_output() -> Iterator[None]:
    """Keep MLflow's optional emoji links from breaking non-UTF-8 terminals.

    MLflow prints run URLs with emoji when a REST tracking store closes a run.
    Windows PowerShell can expose a CP1252 stdout stream, where that otherwise
    successful close raises ``UnicodeEncodeError`` before promotion happens.
    """
    name = "MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"
    previous = os.environ.get(name)
    os.environ[name] = "true"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


class RegistryUnavailable(RuntimeError):
    """MLflow is unreachable, or the requested release does not exist."""


@dataclass(frozen=True)
class ReleaseSpec:
    """Everything that defines one serving release."""

    prompt_version: str
    prompt_template: str
    vllm_model_id: str
    embedding_model_id: str
    qdrant_collection: str
    feature_service: str
    top_k: int
    delta_version: int | None = None
    evaluation: dict[str, float] = field(default_factory=dict)

    def as_params(self) -> dict[str, str]:
        return {
            "prompt_version": self.prompt_version,
            "vllm_model_id": self.vllm_model_id,
            "embedding_model_id": self.embedding_model_id,
            "qdrant_collection": self.qdrant_collection,
            "feature_service": self.feature_service,
            "top_k": str(self.top_k),
            "delta_version": str(self.delta_version),
        }

    def as_tags(self) -> dict[str, str]:
        return {
            TAG_PROMPT_VERSION: self.prompt_version,
            TAG_VLLM_MODEL: self.vllm_model_id,
            TAG_EMBEDDING_MODEL: self.embedding_model_id,
            TAG_DELTA_VERSION: str(self.delta_version),
            TAG_COLLECTION: self.qdrant_collection,
            TAG_FEATURE_SERVICE: self.feature_service,
        }


@dataclass(frozen=True)
class Release:
    """A resolved registry entry the serving path can act on."""

    name: str
    version: str
    run_id: str
    alias: str
    prompt_version: str
    prompt_template: str
    vllm_model_id: str
    embedding_model_id: str
    delta_version: int | None
    top_k: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "run_id": self.run_id,
            "alias": self.alias,
            "prompt_version": self.prompt_version,
            "vllm_model_id": self.vllm_model_id,
            "embedding_model_id": self.embedding_model_id,
            "delta_version": self.delta_version,
            "top_k": self.top_k,
        }


class ReleaseRegistry:
    """Register, promote, roll back and resolve serving releases."""

    def __init__(self, settings: MLflowSettings) -> None:
        self._settings = settings
        mlflow.set_tracking_uri(settings.tracking_uri)
        self._client = MlflowClient(tracking_uri=settings.tracking_uri)

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    # -- registration ------------------------------------------------------

    def register(self, spec: ReleaseSpec, *, promote: bool = False) -> Release:
        """Log an evaluated release and create a registered version for it.

        The artifact is a small pyfunc that renders the prompt. It exists so
        the version carries a real signature and a loadable payload rather
        than being a bare row in a table — the prompt and configuration travel
        with the version, which is what makes rollback meaningful.
        """
        mlflow.set_experiment(self._settings.experiment)
        try:
            with (
                _portable_mlflow_output(),
                mlflow.start_run(run_name=f"release-{spec.prompt_version}") as run,
            ):
                mlflow.log_params(spec.as_params())
                if spec.evaluation:
                    mlflow.log_metrics(spec.evaluation)
                mlflow.log_dict(
                    {"template": spec.prompt_template, "version": spec.prompt_version},
                    "prompt.json",
                )
                info = mlflow.pyfunc.log_model(
                    name="release",
                    python_model=_PromptRelease(),
                    artifacts=None,
                    input_example=["Nền tảng dữ liệu là gì?"],
                    metadata=spec.as_tags() | {"prompt_template": spec.prompt_template},
                    registered_model_name=self._settings.model_name,
                    tags=spec.as_tags(),
                )
                run_id = run.info.run_id
        except MlflowException as error:
            raise RegistryUnavailable(f"could not register release: {error}") from error

        version = self._version_for(info.model_uri, run_id)
        for key, value in spec.as_tags().items():
            self._client.set_model_version_tag(
                self._settings.model_name, version, key, value
            )
        metrics.RELEASE_TRANSITIONS.labels(action="registered").inc()

        if promote:
            return self.promote(version)
        return self._describe(version, alias="")

    def _version_for(self, model_uri: str, run_id: str) -> str:
        """Find the registered version created by this run."""
        versions = self._client.search_model_versions(
            f"name='{self._settings.model_name}' and run_id='{run_id}'"
        )
        if not versions:
            raise RegistryUnavailable(
                f"run {run_id} logged {model_uri} but no registered version appeared"
            )
        return max(versions, key=lambda item: int(item.version)).version

    # -- promotion and rollback -------------------------------------------

    def promote(self, version: str) -> Release:
        """Point the serving alias at ``version``."""
        previous = self.promoted_version()
        try:
            self._client.set_registered_model_alias(
                self._settings.model_name, self._settings.alias, version
            )
        except MlflowException as error:
            raise RegistryUnavailable(f"could not promote {version}: {error}") from error

        action = "rolled_back" if _is_rollback(previous, version) else "promoted"
        metrics.RELEASE_TRANSITIONS.labels(action=action).inc()
        release = self._describe(version, alias=self._settings.alias)
        metrics.set_release(
            model_name=release.name,
            version=release.version,
            run_id=release.run_id,
            vllm_model_id=release.vllm_model_id,
            delta_version=release.delta_version,
        )
        return release

    def rollback(self) -> Release:
        """Move the alias to the highest version below the current one."""
        current = self.current_version()
        candidates = sorted(
            (
                int(item.version)
                for item in self._client.search_model_versions(
                    f"name='{self._settings.model_name}'"
                )
                if int(item.version) < int(current)
            ),
            reverse=True,
        )
        if not candidates:
            raise RegistryUnavailable(
                f"{self._settings.model_name} v{current} is the only version; "
                "there is nothing to roll back to"
            )
        return self.promote(str(candidates[0]))

    def current_version(self) -> str:
        """The version the alias points at, or raise when nothing is promoted.

        Split from :meth:`promoted_version` on purpose: an operation like
        rollback has no meaning without a current release and should say so,
        while promotion of the very first version legitimately finds none.
        """
        version = self.promoted_version()
        if version is None:
            raise RegistryUnavailable(
                f"no '{self._settings.alias}' alias on {self._settings.model_name}"
            )
        return version

    def promoted_version(self) -> str | None:
        """The version the alias points at, or ``None`` when there is no alias."""
        try:
            entry = self._client.get_model_version_by_alias(
                self._settings.model_name, self._settings.alias
            )
        except MlflowException:
            return None
        return entry.version

    # -- resolution --------------------------------------------------------

    def resolve(self) -> Release:
        """Load the release the serving path must use for this request."""
        with span(
            SPAN_MLFLOW_RESOLVE,
            attributes={
                "lab28.registry.model": self._settings.model_name,
                "lab28.registry.alias": self._settings.alias,
            },
        ) as active:
            try:
                entry = self._client.get_model_version_by_alias(
                    self._settings.model_name, self._settings.alias
                )
            except MlflowException as error:
                raise RegistryUnavailable(
                    f"no '{self._settings.alias}' release for "
                    f"{self._settings.model_name}: {error}"
                ) from error
            release = self._from_entry(entry, alias=self._settings.alias)
            active.set_attribute("lab28.registry.version", release.version)
            metrics.set_release(
                model_name=release.name,
                version=release.version,
                run_id=release.run_id,
                vllm_model_id=release.vllm_model_id,
                delta_version=release.delta_version,
            )
            return release

    def _describe(self, version: str, *, alias: str) -> Release:
        try:
            entry = self._client.get_model_version(self._settings.model_name, version)
        except MlflowException as error:
            raise RegistryUnavailable(f"version {version} not found: {error}") from error
        return self._from_entry(entry, alias=alias)

    def _from_entry(self, entry: Any, *, alias: str) -> Release:
        tags = dict(entry.tags or {})
        params = self._run_params(entry.run_id)
        return Release(
            name=entry.name,
            version=entry.version,
            run_id=entry.run_id,
            alias=alias,
            prompt_version=tags.get(TAG_PROMPT_VERSION, params.get("prompt_version", "")),
            prompt_template=self._prompt_template(entry.run_id),
            vllm_model_id=tags.get(TAG_VLLM_MODEL, params.get("vllm_model_id", "")),
            embedding_model_id=tags.get(
                TAG_EMBEDDING_MODEL, params.get("embedding_model_id", "")
            ),
            delta_version=_as_int(tags.get(TAG_DELTA_VERSION)),
            top_k=_as_int(params.get("top_k")) or 3,
        )

    def _run_params(self, run_id: str) -> dict[str, str]:
        try:
            return dict(self._client.get_run(run_id).data.params)
        except MlflowException:
            return {}

    def _prompt_template(self, run_id: str) -> str:
        try:
            path = self._client.download_artifacts(run_id, "prompt.json")
        except MlflowException as error:
            raise RegistryUnavailable(
                f"release run {run_id} has no prompt.json artifact: {error}"
            ) from error
        with open(path, encoding="utf-8") as handle:
            return str(json.load(handle).get("template", ""))

    # -- health ------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Readiness: the registry answers, and a champion release exists."""
        try:
            release = self.resolve()
        except RegistryUnavailable as error:
            return {
                "reachable": _registry_reachable(self._client, self._settings),
                "has_champion": False,
                "detail": str(error),
            }
        return {
            "reachable": True,
            "has_champion": True,
            "version": release.version,
            "run_id": release.run_id,
            "detail": f"{release.name} v{release.version} is {self._settings.alias}",
        }


class _PromptRelease(mlflow.pyfunc.PythonModel):
    """Renders the release prompt. Loadable proof that a version is complete."""

    def load_context(self, context: Any) -> None:
        self._template = (context.model_config or {}).get("prompt_template", "{question}")

    # Annotated on purpose: MLflow reads these hints to infer the model
    # signature, and warns on every import of the serving path when they are
    # missing. ``list[str]`` is one of the hints it can turn into a schema, so
    # the logged version carries a real signature instead of a guess.
    def predict(
        self,
        context: Any,
        model_input: list[str],
        params: dict[str, Any] | None = None,
    ) -> list[str]:
        template = getattr(self, "_template", "{question}")
        return [template.replace("{question}", question) for question in model_input]


def _registry_reachable(client: MlflowClient, settings: MLflowSettings) -> bool:
    try:
        client.search_registered_models(f"name='{settings.model_name}'", max_results=1)
    except MlflowException:
        return False
    return True


def _is_rollback(previous: str | None, target: str) -> bool:
    if not previous:
        return False
    try:
        return int(target) < int(previous)
    except ValueError:
        return False


def _as_int(raw: Any) -> int | None:
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None
