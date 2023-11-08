import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Union
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.trace import SpanKind

from baserun.grpc import (
    get_or_create_submission_service,
    get_or_create_async_submission_service,
)
from baserun.v1.baserun_pb2 import (
    StartSessionRequest,
    Session,
    EndUser,
    EndSessionRequest,
    Run,
)
from . import Baserun
from .instrumentation.span_attributes import SpanAttributes

logger = logging.getLogger(__name__)


@contextmanager
def with_session(
    user_identifier: str, session_identifier: str = None, auto_end: bool = True
):
    tracer_provider = trace.get_tracer_provider()
    tracer = tracer_provider.get_tracer("baserun")

    run = Baserun.current_run()
    session = start_session(
        user_identifier=user_identifier, session_identifier=session_identifier, run=run
    )
    try:
        with tracer.start_as_current_span(
            f"baserun.session",
            kind=SpanKind.CLIENT,
            attributes={
                SpanAttributes.BASERUN_SESSION_ID: session.identifier,
                SpanAttributes.BASERUN_RUN: Baserun.serialize_run(run),
            },
        ):
            yield
    finally:
        if auto_end:
            end_session(session)


def start_session(
    user_identifier: str,
    start_timestamp: datetime = None,
    session_identifier: str = None,
    run: Run = None,
) -> Session:
    session = Session(
        identifier=session_identifier or str(uuid4()),
        end_user=EndUser(identifier=user_identifier),
    )
    session.start_timestamp.FromDatetime(start_timestamp or datetime.utcnow())
    session_request = StartSessionRequest(session=session, run=run)
    try:
        response = get_or_create_submission_service().StartSession(session_request)
        return response.session
    except Exception as e:
        if hasattr(e, "details"):
            logger.warning(f"Failed to submit session to Baserun: {e.details()}")
        else:
            logger.warning(f"Failed to submit session to Baserun: {e}")

        session.id = str(uuid4())
        return session


def end_session(
    session: Union[str, Session],
    completion_timestamp: datetime = None,
):
    if not isinstance(session, Session):
        session = Session(
            identifier=session,
        )

    session.completion_timestamp.FromDatetime(completion_timestamp or datetime.utcnow())
    session_request = EndSessionRequest(session=session)
    try:
        get_or_create_submission_service().EndSession(session_request)
    except Exception as e:
        if hasattr(e, "details"):
            logger.warning(f"Failed to submit session to Baserun: {e.details()}")
        else:
            logger.warning(f"Failed to submit session to Baserun: {e}")


async def astart_session(
    user_identifier: str,
    start_timestamp: datetime = None,
    identifier: str = None,
):
    session = Session(
        identifier=identifier or str(uuid4()),
        end_user=EndUser(identifier=user_identifier),
    )
    session.start_timestamp.FromDatetime(start_timestamp or datetime.utcnow())
    session_request = StartSessionRequest(session=session)
    try:
        await get_or_create_async_submission_service().StartSession(session_request)
    except Exception as e:
        if hasattr(e, "details"):
            logger.warning(f"Failed to submit session to Baserun: {e.details()}")
        else:
            logger.warning(f"Failed to submit session to Baserun: {e}")


async def aend_session(
    identifier: str,
    completion_timestamp: datetime = None,
):
    session = Session(
        identifier=identifier,
    )
    session.completion_timestamp.FromDatetime(completion_timestamp or datetime.utcnow())
    session_request = EndSessionRequest(session=session)
    try:
        await get_or_create_async_submission_service().EndSession(session_request)
    except Exception as e:
        if hasattr(e, "details"):
            logger.warning(f"Failed to submit session to Baserun: {e.details()}")
        else:
            logger.warning(f"Failed to submit session to Baserun: {e}")
