import logging

from baserun.grpc import (
    get_or_create_async_submission_service,
    get_or_create_submission_service,
)
from baserun.v1.baserun_pb2 import EndUser, SubmitUserRequest

logger = logging.getLogger(__name__)


def submit_user(identifier: str):
    from baserun import Baserun

    user = EndUser(identifier=identifier)
    user_request = SubmitUserRequest(user=user)

    try:
        Baserun.add_future(get_or_create_submission_service().SubmitUser.future(user_request))
    except Exception as e:
        if hasattr(e, "details"):
            logger.warning(f"Failed to submit user to Baserun: {e.details()}")
        else:
            logger.warning(f"Failed to submit user to Baserun: {e}")


async def asubmit_user(identifier: str):
    user = EndUser(identifier=identifier)
    user_request = SubmitUserRequest(user=user)
    try:
        await get_or_create_async_submission_service().SubmitUser(user_request)
    except Exception as e:
        if hasattr(e, "details"):
            logger.warning(f"Failed to submit user to Baserun: {e.details()}")
        else:
            logger.warning(f"Failed to submit user to Baserun: {e}")
