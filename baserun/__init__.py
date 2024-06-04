from typing import TYPE_CHECKING, Optional, Type, TypeVar, Union

from baserun.mixins import ClientMixin

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


def init(client: T, name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> ClientMixin | T:
    if is_openai_installed():
        from openai import AsyncOpenAI as BaseAsyncOpenAI
        from openai import OpenAI as BaseOpenAI

        from baserun.wrappers.openai import (
            WrappedAsyncOpenAIClient,
            WrappedSyncOpenAIClient,
        )

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

        global Anthropic
        global AsyncAnthropic
        Anthropic = WrappedAnthropic
        AsyncAnthropic = WrappedAsyncAnthropic

        if isinstance(client, BaseAnthropic):
            return WrappedSyncAnthropicClient(**kwargs, name=name, api_key=api_key, client=client)
        elif isinstance(client, BaseAsyncAnthropic):
            return WrappedAsyncAnthropicClient(**kwargs, name=name, api_key=api_key, client=client)

    setup_globals()
    return client


setup_globals()
