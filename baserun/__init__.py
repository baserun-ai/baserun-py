import os
from .baserun import Baserun
from .unittests_class import BaserunTestCase

init = Baserun.init
test = Baserun.test
trace = Baserun.trace
log = Baserun.log
flush = Baserun.flush
evals = Baserun.evals
api_key = os.environ.get('BASERUN_API_KEY')
