import logging
import tempfile

import tvm
from tvm import tir
import pytest
from tvm.meta_schedule import ReplayTraceConfig, tune_tir
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule import schedule_rule, postproc
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.script import tir as T
from tvm.target.target import Target
from tvm.te.operation import create_prim_func
from tvm.tir import Schedule
from tvm.meta_schedule.testing import te_workload, tir_tensor_intrin

@T.prim_func
def injective(A: T.Buffer[(128, ),"float32"], B: T.Buffer[(128, ),"float32"]):
    for i in T.thread_binding(0, 32, "threadIdx.x"):
        for j in T.vectorized(0, 4):
            if j>=1:
                B[i*4+ j] = T.max(A[i*4+j], T.float32(0))


print(tvm.lower(injective, None, simple_mode=True))
f = tvm.build(injective, target="cuda", name="dense")
print(f.imported_modules[0].get_source())
