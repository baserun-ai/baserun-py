from typing import TYPE_CHECKING, Any, List, Optional, Type, TypeVar, Union

from datasets import Dataset, DatasetInfo

from baserun.api import ApiClient
from baserun.mixins import ClientMixin
from baserun.models.dataset import DatasetMetadata
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

    if is_anthropic_installed():
        from baserun.wrappers.anthropic import Anthropic as WrappedAnthropic
        from baserun.wrappers.anthropic import AsyncAnthropic as WrappedAsyncAnthropic

        global Anthropic
        global AsyncAnthropic
        Anthropic = WrappedAnthropic
        AsyncAnthropic = WrappedAsyncAnthropic
        AsyncOpenAI = WrappedAsyncOpenAI


def init(client: T, name: Optional[str] = None, baserun_api_key: Optional[str] = None, **kwargs) -> ClientMixin | T:
    if is_openai_installed():
        from openai import AsyncOpenAI as BaseAsyncOpenAI
        from openai import OpenAI as BaseOpenAI

        from baserun.wrappers.openai import (
            WrappedAsyncOpenAIClient,
            WrappedSyncOpenAIClient,
        )

        if isinstance(client, BaseAsyncOpenAI):
            return WrappedAsyncOpenAIClient(**kwargs, name=name, baserun_api_key=baserun_api_key, client=client)
        elif isinstance(client, BaseOpenAI):
            return WrappedSyncOpenAIClient(**kwargs, name=name, baserun_api_key=baserun_api_key, client=client)

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
            return WrappedSyncAnthropicClient(**kwargs, name=name, baserun_api_key=baserun_api_key, client=client)
        elif isinstance(client, BaseAsyncAnthropic):
            return WrappedAsyncAnthropicClient(**kwargs, name=name, baserun_api_key=baserun_api_key, client=client)

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


def submit_dataset(dataset: Any, name: str, api_key: Optional[str] = None):
    api_client = ApiClient(api_key=api_key)
    api_client.submit_dataset(dataset, name)


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

    dataset = Dataset.from_list(
        raw_dataset.get("object", []),
        info=DatasetInfo(
            dataset_name=raw_dataset.get("name"),
        ),
    )
    return dataset


setup_globals()
