import os
from .baserun import Baserun

init = Baserun.init
trace = Baserun.trace
log = Baserun.log
evals = Baserun.evals
api_key = os.environ.get('BASERUN_API_KEY')
