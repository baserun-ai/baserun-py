import pytest

from baserun import asubmit_user, submit_user


def test_submit_user(mock_services):
    user_identifier = "foo@test.com"
    submit_user(user_identifier)

    mock_submit_user = mock_services["submission_service"].SubmitUser.future
    assert mock_submit_user.call_count == 1
    args, kwargs = mock_submit_user.call_args_list[0]

    submit_user_request = args[0]
    assert submit_user_request.user.identifier == user_identifier


@pytest.mark.asyncio
async def test_asubmit_user(mock_services):
    user_identifier = "foo@test.com"
    await asubmit_user(user_identifier)

    mock_submit_user = mock_services["async_submission_service"].SubmitUser
    assert mock_submit_user.call_count == 1
    args, kwargs = mock_submit_user.call_args_list[0]

    submit_user_request = args[0]
    assert submit_user_request.user.identifier == user_identifier
