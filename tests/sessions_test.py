from uuid import uuid4

import pytest

from baserun import aend_session, astart_session, end_session, start_session


def test_start_session(mock_services):
    user_identifier = "foo@test.com"
    start_session(user_identifier)

    mock_start_session = mock_services["submission_service"].StartSession
    assert mock_start_session.call_count == 1
    args, kwargs = mock_start_session.call_args_list[0]

    start_session_request = args[0]
    assert start_session_request.session.end_user.identifier == user_identifier
    assert len(start_session_request.session.identifier) == 36
    assert start_session_request.session.start_timestamp


def test_end_session(mock_services):
    session_identifier = str(uuid4())
    end_session(session_identifier)

    mock_end_session = mock_services["submission_service"].EndSession.future
    assert mock_end_session.call_count == 1
    args, kwargs = mock_end_session.call_args_list[0]

    end_session_request = args[0]
    assert end_session_request.session.identifier == session_identifier
    assert end_session_request.session.completion_timestamp


@pytest.mark.asyncio
async def test_astart_session(mock_services):
    user_identifier = "foo@test.com"
    await astart_session(user_identifier)

    mock_start_session = mock_services["async_submission_service"].StartSession
    assert mock_start_session.call_count == 1
    args, kwargs = mock_start_session.call_args_list[0]

    start_session_request = args[0]
    assert start_session_request.session.end_user.identifier == user_identifier
    assert len(start_session_request.session.identifier) == 36
    assert start_session_request.session.start_timestamp


@pytest.mark.asyncio
async def test_aend_session(mock_services):
    session_identifier = str(uuid4())
    await aend_session(session_identifier)

    mock_end_session = mock_services["async_submission_service"].EndSession
    assert mock_end_session.call_count == 1
    args, kwargs = mock_end_session.call_args_list[0]

    end_session_request = args[0]
    assert end_session_request.session.identifier == session_identifier
    assert end_session_request.session.completion_timestamp
