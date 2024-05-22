from typing import Optional, TypeVar

from baserun.mixins import ClientMixin

T = TypeVar("T")

try:
    from openai import AsyncOpenAI as BaseAsyncOpenAI

    from baserun.wrappers.openai import AsyncOpenAI as WrappedAsyncOpenAI
    from baserun.wrappers.openai import OpenAI as WrappedOpenAI
    from baserun.wrappers.openai import (
        WrappedAsyncOpenAIClient,
        WrappedOpenAIBaseClient,
        WrappedSyncOpenAIClient,
    )

    OpenAI = WrappedOpenAI
    AsyncOpenAI = WrappedAsyncOpenAI

    def init(client: T, name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> WrappedOpenAIBaseClient:
        if isinstance(client, BaseAsyncOpenAI):
            return WrappedAsyncOpenAIClient(**kwargs, name=name, api_key=api_key, client=client)
        return WrappedSyncOpenAIClient(**kwargs, name=name, api_key=api_key, client=client)

except ImportError:
    # TODO: Handle anthropic and other providers here (in case they've installed any of those alongside openai)

    class BaseOpenAI:
        pass

    class WrappedSyncOpenAIClient(ClientMixin):  # type: ignore
        pass

    class WrappedAsyncOpenAIClient(ClientMixin):  # type: ignore
        pass

    def init(client: T, name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> WrappedOpenAIBaseClient:
        raise ImportError("No supported libraries are installed.")
