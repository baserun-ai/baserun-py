import logging
from queue import Queue, Empty
from time import sleep

from baserun.v1.baserun_pb2 import (
    SubmitSpanRequest,
)
from .grpc import get_or_create_submission_service

logger = logging.getLogger(__name__)


def worker(queue: Queue):
    submission_service = get_or_create_submission_service()
    while True:
        try:
            item: SubmitSpanRequest = queue.get(timeout=1)
        except Empty:
            continue

        if item is None:
            break

        try:
            result = submission_service.SubmitSpan(item)
            logger.debug(f"Submitted {item} and got {result}")
        except Exception as e:
            if hasattr(e, "details"):
                # Race condition where the span is submitted before the run start call finishes
                if "not found" in e.details():
                    sleep(0.5)
                    submission_service = get_or_create_submission_service()
                    submission_service.SubmitSpan(item)
                else:
                    logger.warning(f"Failed to submit span to Baserun: {e.details()}")
            else:
                logger.warning(f"Failed to submit span to Baserun: {e}")
        finally:
            queue.task_done()
