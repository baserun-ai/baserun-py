from typing import TYPE_CHECKING, Optional, TypeVar, Union

from baserun.mixins import ClientMixin

if TYPE_CHECKING:
    try:
        import openai
    except ImportError:
        pass

    try:
        import anthropic
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


OpenAI: Union["openai.OpenAI", None] = None
AsyncOpenAI: Union["openai.AsyncOpenAI", None] = None
Anthropic: Union["anthropic.Anthropic", None] = None
AsyncAnthropic: Union["anthropic.AsyncAnthropic", None] = None


def init(client: T, name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> ClientMixin | T:
    if is_openai_installed():
        from openai import AsyncOpenAI as BaseAsyncOpenAI
        from openai import OpenAI as BaseOpenAI

        from baserun.wrappers.openai import AsyncOpenAI as WrappedAsyncOpenAI
        from baserun.wrappers.openai import OpenAI as WrappedOpenAI
        from baserun.wrappers.openai import (
            WrappedAsyncOpenAIClient,
            WrappedSyncOpenAIClient,
        )

        OpenAI = WrappedOpenAI  # noqa: F841
        AsyncOpenAI = WrappedAsyncOpenAI  # noqa: F841

        if isinstance(client, BaseAsyncOpenAI):
            return WrappedAsyncOpenAIClient(**kwargs, name=name, api_key=api_key, client=client)
        elif isinstance(client, BaseOpenAI):
            return WrappedSyncOpenAIClient(**kwargs, name=name, api_key=api_key, client=client)

    if is_anthropic_installed():
        from anthropic import Anthropic as BaseAnthropic
        from anthropic import AsyncAnthropic as BaseAsyncAnthropic

        from baserun.wrappers.anthropic import Anthropic as WrappedAnthropic
        from baserun.wrappers.anthropic import AsyncAnthropic as WrappedAsyncAnthropic
        from baserun.wrappers.anthropic import (
            WrappedAsyncAnthropicClient,
            WrappedSyncAnthropicClient,
        )

        Anthropic = WrappedAnthropic  # noqa: F841
        AsyncAnthropic = WrappedAsyncAnthropic  # noqa: F841

        if isinstance(client, BaseAnthropic):
            return WrappedSyncAnthropicClient(**kwargs, name=name, api_key=api_key, client=client)
        elif isinstance(client, BaseAsyncAnthropic):
            return WrappedAsyncAnthropicClient(**kwargs, name=name, api_key=api_key, client=client)

    return client
