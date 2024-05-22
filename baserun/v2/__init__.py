from typing import Any, Optional, TypeVar, Union

from baserun.v2.mixins import ClientMixin

T = TypeVar("T")

try:
    from openai import AsyncOpenAI as BaseAsyncOpenAI

    from baserun.v2.wrappers.openai import (
        OpenAI as WrappedOpenAI,
        AsyncOpenAI as WrappedAsyncOpenAI,
        WrappedOpenAIBaseClient,
        WrappedSyncOpenAIClient,
        WrappedAsyncOpenAIClient,
    )

    OpenAI = WrappedOpenAI
    AsyncOpenAI = WrappedAsyncOpenAI

    def init(client: T, name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> WrappedOpenAIBaseClient:
        if isinstance(client, BaseAsyncOpenAI):
            return WrappedAsyncOpenAIClient(**kwargs, name=name, api_key=api_key, client=client)
        return WrappedSyncOpenAIClient(**kwargs, name=name, api_key=api_key, client=client)

except ImportError as e:
    import pdb

    pdb.set_trace()

    # TODO: Handle anthropic and other providers here (in case they've installed any of those alongside openai)

    class BaseOpenAI:
        pass

    class WrappedSyncOpenAIClient(ClientMixin):  # type: ignore
        pass

    class WrappedAsyncOpenAIClient(ClientMixin):  # type: ignore
        pass

    def init(client: T, name: Optional[str] = None, api_key: Optional[str] = None, **kwargs) -> WrappedOpenAIBaseClient:
        raise ImportError("No supported libraries are installed.")
