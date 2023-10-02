import os

from .baserun import Baserun
from .thread_wrapper import baserun_thread_wrapper

init = Baserun.init
trace = Baserun.trace
log = Baserun.log
evals = Baserun.evals
thread_wrapper = baserun_thread_wrapper
api_key = os.environ.get("BASERUN_API_KEY")
