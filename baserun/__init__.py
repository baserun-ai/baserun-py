import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union
from uuid import uuid4

from datasets import Dataset, DatasetInfo
from datasets.fingerprint import generate_fingerprint

from baserun.api import ApiClient
from baserun.mixins import ClientMixin, CompletionMixin
from baserun.models.dataset import DatasetMetadata
from baserun.models.evals import CompletionEval, Eval, TraceEval
from baserun.models.evaluators import Evaluator
from baserun.models.experiment import Experiment
from baserun.models.scenario import Scenario
from baserun.models.tags import Tag
from baserun.wrappers.generic import GenericClient, GenericCompletion

if TYPE_CHECKING:
    try:
        from baserun.wrappers import openai
    except ImportError:
        pass

    try:
        from baserun.wrappers import anthropic
    except ImportError:
        pass

T = TypeVar("T")


def is_openai_installed() -> bool:
    try:
        import openai  # noqa: F401

        return True
    except ImportError:
        return False


def is_anthropic_installed() -> bool:
    try:
        import anthropic  # noqa: F401

        return True
    except ImportError:
        return False


class BaseOpenAI:
    pass


class WrappedSyncOpenAIClient(ClientMixin):  # type: ignore
    pass


class WrappedAsyncOpenAIClient(ClientMixin):  # type: ignore
    pass


OpenAI: Union[Type["openai.OpenAI"], None] = None
AsyncOpenAI: Union[Type["openai.AsyncOpenAI"], None] = None
Anthropic: Union[Type["anthropic.Anthropic"], None] = None
AsyncAnthropic: Union[Type["anthropic.AsyncAnthropic"], None] = None


def setup_globals():
    if is_openai_installed():
        from baserun.wrappers.openai import AsyncOpenAI as WrappedAsyncOpenAI
        from baserun.wrappers.openai import OpenAI as WrappedOpenAI

        global OpenAI
        global AsyncOpenAI
        OpenAI = WrappedOpenAI
        AsyncOpenAI = WrappedAsyncOpenAI

    if is_anthropic_installed():
        from baserun.wrappers.anthropic import Anthropic as WrappedAnthropic
        from baserun.wrappers.anthropic import AsyncAnthropic as WrappedAsyncAnthropic

        global Anthropic
        global AsyncAnthropic
        Anthropic = WrappedAnthropic
        AsyncAnthropic = WrappedAsyncAnthropic


def init(
    client: T,
    name: Optional[str] = None,
    baserun_api_key: Optional[str] = None,
    experiment: Optional[Experiment] = None,
    **kwargs,
) -> ClientMixin | T:
    if is_openai_installed():
        from openai import AsyncOpenAI as BaseAsyncOpenAI
        from openai import OpenAI as BaseOpenAI

        from baserun.wrappers.openai import (
            WrappedAsyncOpenAIClient,
            WrappedSyncOpenAIClient,
        )

        if isinstance(client, BaseAsyncOpenAI):
            return WrappedAsyncOpenAIClient(
                **kwargs,
                name=name,
                baserun_api_key=baserun_api_key,
                client=client,
                experiment=experiment,
            )
        elif isinstance(client, BaseOpenAI):
            return WrappedSyncOpenAIClient(
                **kwargs,
                name=name,
                baserun_api_key=baserun_api_key,
                client=client,
                experiment=experiment,
            )

    if is_anthropic_installed():
        from anthropic import Anthropic as BaseAnthropic
        from anthropic import AsyncAnthropic as BaseAsyncAnthropic

        from baserun.wrappers.anthropic import Anthropic as WrappedAnthropic
        from baserun.wrappers.anthropic import AsyncAnthropic as WrappedAsyncAnthropic
        from baserun.wrappers.anthropic import (
            WrappedAsyncAnthropicClient,
            WrappedSyncAnthropicClient,
        )

        global Anthropic
        global AsyncAnthropic
        Anthropic = WrappedAnthropic
        AsyncAnthropic = WrappedAsyncAnthropic

        if isinstance(client, BaseAnthropic):
            return WrappedSyncAnthropicClient(
                **kwargs,
                name=name,
                baserun_api_key=baserun_api_key,
                client=client,
                experiment=experiment,
            )
        elif isinstance(client, BaseAsyncAnthropic):
            return WrappedAsyncAnthropicClient(
                **kwargs,
                name=name,
                baserun_api_key=baserun_api_key,
                client=client,
                experiment=experiment,
            )

    setup_globals()
    return client


def tag(
    key: str,
    value: str,
    trace_id: str,
    completion_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    tag_type: Optional[str] = "custom",
) -> Tag:
    api_client = ApiClient()
    generic_client = GenericClient(trace_id=trace_id, name="resumed", integrations=[], api_client=api_client)

    if completion_id:
        generic_completion = GenericCompletion(
            name="resumed",
            client=generic_client,
            completion_id=completion_id,
            trace_id=trace_id,
        )
        return generic_completion.tag(key, value, metadata=metadata, tag_type=tag_type)
    else:
        return generic_client.tag(key, value, metadata=metadata, tag_type=tag_type)


def evaluate(
    evaluators: List[Evaluator],
    scenario: Scenario,
    name: Optional[str] = None,
    score: Optional[float] = None,
    trace_id: Optional[str] = None,
    client: Optional[ClientMixin] = None,
    completion_id: Optional[str] = None,
    completion: Optional[Union[GenericCompletion, CompletionMixin]] = None,
    metadata: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> List[Union[CompletionEval, TraceEval, Eval]]:
    client = client or completion.client if completion else client
    client = client or scenario.client if scenario.client else client
    if not client and not trace_id:
        raise ValueError("Either client or trace_id must be set")

    metadata = metadata or scenario.metadata or {}

    name = name or "resumed"

    if not client and trace_id:
        client = GenericClient(trace_id=trace_id, name=name, api_client=ApiClient(api_key=api_key), metadata=metadata)

    if client and not completion and completion_id:
        completion = GenericCompletion(
            completion_id=completion_id, client=client.genericize(), name=name, metadata=metadata
        )

    evaluations: List[Union[Eval, TraceEval, CompletionEval]] = []
    if completion:
        for evaluator in evaluators:
            if score is None:
                completion_eval = evaluator.evaluate()
            else:
                completion_eval = evaluator._create_completion_eval(score=score, metadata=metadata)

            if completion_eval:
                evaluations.append(completion_eval)
        completion.submit_to_baserun()
    elif client:
        for evaluator in evaluators:
            trace_eval: Optional[Union[Eval, TraceEval]] = None
            if score:
                trace_eval = evaluator._create_trace_eval(score=score, metadata=metadata)
            else:
                trace_eval = evaluator.evaluate()

            if trace_eval:
                evaluations.append(trace_eval)
        client.submit_to_baserun()

    return evaluations


def log(
    message: str,
    trace_id: str,
    name: Optional[str] = None,
    metadata: Optional[dict] = None,
    completion_id: Optional[str] = None,
) -> Tag:
    return tag(
        key=name or "log",
        value=message,
        trace_id=trace_id,
        completion_id=completion_id,
        metadata=metadata,
        tag_type="log",
    )


def variable(
    key: str,
    value: str,
    trace_id: str,
    completion_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Tag:
    return tag(key, value, trace_id, completion_id, metadata, tag_type="variable")


def feedback(
    name: str,
    score: Any,
    trace_id: str,
    completion_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Tag:
    return tag(name, score, trace_id, completion_id, metadata, tag_type="feedback")


def submit_dataset(
    dataset: Dataset,
    name: Optional[str] = None,
    metadata: Optional[dict] = None,
    version: Optional[str] = None,
    api_key: Optional[str] = None,
):
    # TODO: Validation
    # TODO: Pull in dataset info into metadata (and put it back when fetching)
    if not name:
        name = dataset.info.dataset_name
    else:
        dataset.info.dataset_name = name

    if not name:
        # FIXME? Is this a reasonable default?
        name = str(uuid4())

    metadata = metadata or {}
    for info_key, info_value in dataset.info.__dict__.items():
        # Skip if the attribute is not serializable
        try:
            json.dumps(info_value)
            if info_value:
                metadata[info_key] = info_value
        except TypeError:
            pass

    dataset._fingerprint = generate_fingerprint(dataset)
    api_client = ApiClient(api_key=api_key)
    api_client.submit_dataset(
        dataset.to_list(),
        name,
        metadata=metadata,
        version=version or dataset._fingerprint,
        fingerprint=dataset._fingerprint,
    )


async def list_datasets(api_key: Optional[str] = None) -> List[DatasetMetadata]:
    api_client = ApiClient(api_key=api_key)
    return await api_client.list_datasets()


async def get_dataset(
    id: Optional[str] = None, name: Optional[str] = None, version: Optional[str] = None, api_key: Optional[str] = None
) -> Union[Dataset, None]:
    if not id and not name:
        raise ValueError("Either id or name must be provided")

    api_client = ApiClient(api_key=api_key)
    raw_dataset = await api_client.get_dataset(id=id or name or "", version=version)
    if not raw_dataset:
        return None

    info = DatasetInfo(dataset_name=raw_dataset.get("name"))
    for key, value in raw_dataset.get("metadata", {}).items():
        if hasattr(info, key):
            setattr(info, key, value)

    dataset_object = raw_dataset.get("object", {})
    if isinstance(dataset_object, Dict):
        return Dataset.from_dict(dataset_object, info=info)
    return Dataset.from_list(dataset_object, info=info)


def create_dataset(data: Union[Dict[str, List[Any]], List[Dict[str, Any]]], name: Optional[str] = None) -> Dataset:
    if isinstance(data, Dict):
        data = [data]

    if not name:
        name = str(uuid4())

    return Dataset.from_list(data, info=DatasetInfo(dataset_name=name))


setup_globals()
