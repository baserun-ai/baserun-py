# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from baserun.v1 import baserun_pb2 as v1_dot_baserun__pb2


class SubmissionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.StartRun = channel.unary_unary(
            "/baserun.v1.SubmissionService/StartRun",
            request_serializer=v1_dot_baserun__pb2.StartRunRequest.SerializeToString,
            response_deserializer=v1_dot_baserun__pb2.StartRunResponse.FromString,
        )
        self.SubmitLog = channel.unary_unary(
            "/baserun.v1.SubmissionService/SubmitLog",
            request_serializer=v1_dot_baserun__pb2.SubmitLogRequest.SerializeToString,
            response_deserializer=v1_dot_baserun__pb2.SubmitLogResponse.FromString,
        )
        self.SubmitSpan = channel.unary_unary(
            "/baserun.v1.SubmissionService/SubmitSpan",
            request_serializer=v1_dot_baserun__pb2.SubmitSpanRequest.SerializeToString,
            response_deserializer=v1_dot_baserun__pb2.SubmitSpanResponse.FromString,
        )
        self.EndRun = channel.unary_unary(
            "/baserun.v1.SubmissionService/EndRun",
            request_serializer=v1_dot_baserun__pb2.EndRunRequest.SerializeToString,
            response_deserializer=v1_dot_baserun__pb2.EndRunResponse.FromString,
        )
        self.SubmitEval = channel.unary_unary(
            "/baserun.v1.SubmissionService/SubmitEval",
            request_serializer=v1_dot_baserun__pb2.SubmitEvalRequest.SerializeToString,
            response_deserializer=v1_dot_baserun__pb2.SubmitEvalResponse.FromString,
        )
        self.StartTestSuite = channel.unary_unary(
            "/baserun.v1.SubmissionService/StartTestSuite",
            request_serializer=v1_dot_baserun__pb2.StartTestSuiteRequest.SerializeToString,
            response_deserializer=v1_dot_baserun__pb2.StartTestSuiteResponse.FromString,
        )
        self.EndTestSuite = channel.unary_unary(
            "/baserun.v1.SubmissionService/EndTestSuite",
            request_serializer=v1_dot_baserun__pb2.EndTestSuiteRequest.SerializeToString,
            response_deserializer=v1_dot_baserun__pb2.EndTestSuiteResponse.FromString,
        )


class SubmissionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def StartRun(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SubmitLog(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SubmitSpan(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def EndRun(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def SubmitEval(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def StartTestSuite(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def EndTestSuite(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_SubmissionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "StartRun": grpc.unary_unary_rpc_method_handler(
            servicer.StartRun,
            request_deserializer=v1_dot_baserun__pb2.StartRunRequest.FromString,
            response_serializer=v1_dot_baserun__pb2.StartRunResponse.SerializeToString,
        ),
        "SubmitLog": grpc.unary_unary_rpc_method_handler(
            servicer.SubmitLog,
            request_deserializer=v1_dot_baserun__pb2.SubmitLogRequest.FromString,
            response_serializer=v1_dot_baserun__pb2.SubmitLogResponse.SerializeToString,
        ),
        "SubmitSpan": grpc.unary_unary_rpc_method_handler(
            servicer.SubmitSpan,
            request_deserializer=v1_dot_baserun__pb2.SubmitSpanRequest.FromString,
            response_serializer=v1_dot_baserun__pb2.SubmitSpanResponse.SerializeToString,
        ),
        "EndRun": grpc.unary_unary_rpc_method_handler(
            servicer.EndRun,
            request_deserializer=v1_dot_baserun__pb2.EndRunRequest.FromString,
            response_serializer=v1_dot_baserun__pb2.EndRunResponse.SerializeToString,
        ),
        "SubmitEval": grpc.unary_unary_rpc_method_handler(
            servicer.SubmitEval,
            request_deserializer=v1_dot_baserun__pb2.SubmitEvalRequest.FromString,
            response_serializer=v1_dot_baserun__pb2.SubmitEvalResponse.SerializeToString,
        ),
        "StartTestSuite": grpc.unary_unary_rpc_method_handler(
            servicer.StartTestSuite,
            request_deserializer=v1_dot_baserun__pb2.StartTestSuiteRequest.FromString,
            response_serializer=v1_dot_baserun__pb2.StartTestSuiteResponse.SerializeToString,
        ),
        "EndTestSuite": grpc.unary_unary_rpc_method_handler(
            servicer.EndTestSuite,
            request_deserializer=v1_dot_baserun__pb2.EndTestSuiteRequest.FromString,
            response_serializer=v1_dot_baserun__pb2.EndTestSuiteResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "baserun.v1.SubmissionService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class SubmissionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def StartRun(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/baserun.v1.SubmissionService/StartRun",
            v1_dot_baserun__pb2.StartRunRequest.SerializeToString,
            v1_dot_baserun__pb2.StartRunResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def SubmitLog(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/baserun.v1.SubmissionService/SubmitLog",
            v1_dot_baserun__pb2.SubmitLogRequest.SerializeToString,
            v1_dot_baserun__pb2.SubmitLogResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def SubmitSpan(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/baserun.v1.SubmissionService/SubmitSpan",
            v1_dot_baserun__pb2.SubmitSpanRequest.SerializeToString,
            v1_dot_baserun__pb2.SubmitSpanResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def EndRun(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/baserun.v1.SubmissionService/EndRun",
            v1_dot_baserun__pb2.EndRunRequest.SerializeToString,
            v1_dot_baserun__pb2.EndRunResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def SubmitEval(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/baserun.v1.SubmissionService/SubmitEval",
            v1_dot_baserun__pb2.SubmitEvalRequest.SerializeToString,
            v1_dot_baserun__pb2.SubmitEvalResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def StartTestSuite(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/baserun.v1.SubmissionService/StartTestSuite",
            v1_dot_baserun__pb2.StartTestSuiteRequest.SerializeToString,
            v1_dot_baserun__pb2.StartTestSuiteResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def EndTestSuite(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/baserun.v1.SubmissionService/EndTestSuite",
            v1_dot_baserun__pb2.EndTestSuiteRequest.SerializeToString,
            v1_dot_baserun__pb2.EndTestSuiteResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
