import os

from .baserun import Baserun
from .checks import (
    check as check_method,
    check_equals as check_equals_method,
    check_includes as check_includes_method,
)
from .sessions import (
    start_session as start_session_method,
    end_session as end_session_method,
    with_session as with_session_method,
    astart_session as astart_session_method,
    aend_session as aend_session_method,
)
from .templates import (
    format_prompt as format_prompt_method,
    register_template as register_template_method,
    aformat_prompt as aformat_prompt_method,
    aregister_template as aregister_template_method,
    get_templates as get_templates_method,
    get_template as get_template_method,
)
from .thread_wrapper import baserun_thread_wrapper
from .users import (
    submit_user as submit_user_method,
    asubmit_user as asubmit_user_method,
)

init = Baserun.init
start_trace = Baserun.start_trace
trace = Baserun.trace
log = Baserun.log
evals = Baserun.evals
thread_wrapper = baserun_thread_wrapper
with_session = with_session_method
start_session = start_session_method
end_session = end_session_method
astart_session = astart_session_method
aend_session = aend_session_method
format_prompt = format_prompt_method
register_template = register_template_method
get_template = get_template_method
get_templates = get_templates_method
aformat_prompt = aformat_prompt_method
aregister_template = aregister_template_method
submit_user = submit_user_method
asubmit_user = asubmit_user_method
check = check_method
check_equals = check_equals_method
check_includes = check_includes_method

api_key = os.environ.get("BASERUN_API_KEY")
