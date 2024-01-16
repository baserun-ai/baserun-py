import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Union
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.context import set_value
from opentelemetry.trace import get_current_span

from baserun.grpc import (
    get_or_create_submission_service,
    get_or_create_async_submission_service,
)
from baserun.v1.baserun_pb2 import (
    StartSessionRequest,
    Session,
    EndUser,
    EndSessionRequest,
)
from . import Baserun
from .constants import UNTRACED_SPAN_PARENT_NAME
from .instrumentation.span_attributes import BASERUN_SESSION_ID, BASERUN_USER_ID

logger = logging.getLogger(__name__)


@contextmanager
def with_session(user_identifier: str, session_identifier: str = None, auto_end: bool = True):
    # If there's a current span, start the session in that context. Otherwise, create a parent span
    current_span = get_current_span()
    if current_span.is_recording():
        session = start_session(user_identifier=user_identifier, session_identifier=session_identifier)
        try:
            yield
        finally:
            if auto_end:
                end_session(session)

    else:
        tracer_provider = trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer("baserun")
        old_context = Baserun.get_context()
        with tracer.start_as_current_span(name=UNTRACED_SPAN_PARENT_NAME):
            Baserun.propagate_context(old_context)
            session = start_session(user_identifier=user_identifier, session_identifier=session_identifier)
            try:
                yield
            finally:
                if auto_end:
                    end_session(session)


def start_session(
    user_identifier: str,
    start_timestamp: datetime = None,
    session_identifier: str = None,
) -> Session:
    """Start a session without a context manager. If this function is called directly:
    - A current span must be active
    - `end_session` must be called
    """
    end_user = EndUser(identifier=user_identifier)
    session = Session(identifier=session_identifier or str(uuid4()), end_user=end_user)
    session.start_timestamp.FromDatetime(start_timestamp or datetime.utcnow())

    session_request = StartSessionRequest(session=session)
    try:
        response = get_or_create_submission_service().StartSession(session_request)
        session.id = response.session.id

        if not Baserun.sessions:
            Baserun.sessions = {}
        Baserun.sessions[session.id] = end_user

        # If they're already in a trace go ahead and attach the session to it
        run = Baserun.current_run()
        if run:
            run.session_id = session.id

        # Set the session and user info on the span context
        current_span = get_current_span()
        current_span.set_attribute(BASERUN_SESSION_ID, session.id)
        current_span.set_attribute(BASERUN_USER_ID, user_identifier)
        Baserun.set_context(set_value("session", session, Baserun.get_context()))
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
        get_or_create_submission_service().EndSession.future(session_request)
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
