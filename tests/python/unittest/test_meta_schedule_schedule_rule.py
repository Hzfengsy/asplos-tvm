# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-module-docstring,missing-function-docstring,missing-class-docstring
import math
import re
from typing import List

import tvm
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.schedule_rule import PyScheduleRule
from tvm.script import tir as T
from tvm.tir.schedule import BlockRV, Schedule


# pylint: disable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument,
# fmt: off

@tvm.script.ir_module
class Matmul:
    @T.prim_func
    def main(a: T.handle, b: T.handle, c: T.handle) -> None:
        T.func_attr({"global_symbol": "main"})
        A = T.match_buffer(a, (1024, 1024), "float32")
        B = T.match_buffer(b, (1024, 1024), "float32")
        C = T.match_buffer(c, (1024, 1024), "float32")
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

# fmt: on
# pylint: enable=invalid-name,no-member,line-too-long,too-many-nested-blocks,no-self-argument


def _check_correct(schedule: Schedule):
    trace = schedule.trace
    for inst in trace.decisions:
        assert math.prod(trace.decisions[inst]) == 1024


def test_meta_schedule_schedule_rule():
    class FancyScheduleRule(PyScheduleRule):
        def initialize_with_tune_context(self, tune_context: TuneContext) -> None:
            pass

        def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
            i, j, k = sch.get_loops(block=block)
            i_0, i_1, i_2, i_3 = sch.split(loop=i, factors=[2, 4, 64, 2])
            j_0, j_1, j_2, j_3 = sch.split(loop=j, factors=[4, 64, 2, 2])
            k_0, k_1 = sch.split(loop=k, factors=[32, 32])
            sch.reorder(i_0, j_0, i_1, j_1, k_0, i_2, j_2, k_1, i_3, j_3)
            return [sch]

    sch_rule = FancyScheduleRule()
    mod = Matmul
    sch = Schedule(mod)
    res = sch_rule.apply(sch, block=sch.get_block("matmul"))
    assert len(res) == 1
    try:
        tvm.ir.assert_structural_equal(mod, res[0].mod)
        raise Exception("The schedule rule did not change the schedule.")
    except ValueError:
        _check_correct(res[0])


def test_meta_schedule_schedule_rule_as_string():
    class YetStillSomeFancyScheduleRule(PyScheduleRule):
        def initialize_with_tune_context(self, tune_context: TuneContext) -> None:
            pass

        def apply(self, sch: Schedule, block: BlockRV) -> List[Schedule]:
            pass

    sch_rule = YetStillSomeFancyScheduleRule()
    pattern = re.compile(r"YetStillSomeFancyScheduleRule\(0x[a-f|0-9]*\)")
    assert pattern.match(str(sch_rule))


if __name__ == "__main__":
    test_meta_schedule_schedule_rule()
    test_meta_schedule_schedule_rule_as_string()
