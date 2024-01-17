import os

from .baserun import Baserun
from .checks import (
    check as check,
    check_equals as check_equals,
    check_includes as check_includes,
)
from .sessions import (
    start_session as start_session,
    end_session as end_session,
    with_session as with_session,
    astart_session as astart_session,
    aend_session as aend_session,
)
from .templates import (
    format_prompt as format_prompt,
    register_template as register_template,
    aregister_template as aregister_template,
    get_templates as get_templates,
    get_template as get_template,
)
from .thread_wrapper import baserun_thread_wrapper as thread_wrapper
from .users import (
    submit_user as submit_user,
    asubmit_user as asubmit_user,
)

init = Baserun.init
start_trace = Baserun.start_trace
trace = Baserun.trace
log = Baserun.log
evals = Baserun.evals
api_key = os.environ.get("BASERUN_API_KEY")
annotate = Baserun.annotate

__all__ = [
    "Baserun",
    "api_key",
    "init",
    "start_trace",
    "trace",
    "annotate",
    "log",
    "evals",
    "thread_wrapper",
    "with_session",
    "start_session",
    "end_session",
    "astart_session",
    "aend_session",
    "format_prompt",
    "register_template",
    "get_templates",
    "get_template",
    "aregister_template",
    "submit_user",
    "asubmit_user",
    "check",
    "check_equals",
    "check_includes",
]
