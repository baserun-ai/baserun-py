import os

import grpc

from .api_key import get_api_key
from .v1.baserun_pb2_grpc import SubmissionServiceStub


def credentials() -> grpc.ChannelCredentials:
    if key_chain := os.environ.get("SSL_KEY_CHAIN"):
        ssl_creds = grpc.ssl_channel_credentials(
            root_certificates=bytes(key_chain, "utf-8")
        )
    else:
        ssl_creds = grpc.ssl_channel_credentials()

    call_credentials = grpc.access_token_call_credentials(get_api_key())
    return grpc.composite_channel_credentials(ssl_creds, call_credentials)


def get_or_create_submission_service() -> SubmissionServiceStub:
    from baserun import Baserun

    if Baserun.submission_service is None:
        grpc_base = os.environ.get("BASERUN_GRPC_URI", "grpc.baserun.ai:50051")
        grpc_channel = grpc.secure_channel(grpc_base, credentials())
        Baserun.submission_service = SubmissionServiceStub(grpc_channel)

    return Baserun.submission_service


def get_or_create_async_submission_service() -> SubmissionServiceStub:
    from baserun import Baserun

    if Baserun.async_submission_service is None:
        grpc_base = os.environ.get("BASERUN_GRPC_URI", "grpc.baserun.ai:50051")
        grpc_channel = grpc.aio.secure_channel(grpc_base, credentials())
        Baserun.async_submission_service = SubmissionServiceStub(grpc_channel)

    return Baserun.async_submission_service
