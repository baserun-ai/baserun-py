import grpc
import os
from typing import Union
from .v1.baserun_pb2_grpc import SubmissionServiceStub
from .api_key import get_api_key


submission_service: Union[SubmissionServiceStub, None] = None


def get_or_create_submission_service():
    global submission_service
    if submission_service:
        return submission_service

    if key_chain := os.environ.get("SSL_KEY_CHAIN"):
        ssl_creds = grpc.ssl_channel_credentials(
            root_certificates=bytes(key_chain, "utf-8")
        )
    else:
        ssl_creds = grpc.ssl_channel_credentials()

    call_credentials = grpc.access_token_call_credentials(get_api_key())
    channel_credentials = grpc.composite_channel_credentials(
        ssl_creds, call_credentials
    )
    grpc_base = os.environ.get("BASERUN_GRPC_URI", "grpc.baserun.ai:50051")
    grpc_channel = grpc.secure_channel(grpc_base, channel_credentials)
    submission_service = SubmissionServiceStub(grpc_channel)
    return submission_service


