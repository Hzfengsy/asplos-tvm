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
from typing import List, Union

from tvm.ir import IRModule
from tvm.meta_schedule import TuneContext
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.target import Target
from tvm.tir import PrimFunc, Schedule
from tvm.tir.schedule import Trace

from . import schedule_rule as sch_rule


def create_context(mod: Union[IRModule, PrimFunc], target: Target) -> TuneContext:
    ctx = TuneContext(
        mod=mod,
        target=target,
        space_generator=PostOrderApply(),
        sch_rules=sch_rule.get(target),
        task_name="test",
    )
    ctx.space_generator.initialize_with_tune_context(ctx)
    for rule in ctx.sch_rules:
        rule.initialize_with_tune_context(ctx)
    return ctx


def check_trace(spaces: List[Schedule], expected: List[List[str]]):
    expected_traces = {"\n".join(t) for t in expected}
    actual_traces = set()
    for space in spaces:
        trace = Trace(space.trace.insts, {})
        trace = trace.simplified(remove_postproc=True)
        str_trace = "\n".join(str(trace).strip().splitlines())
        actual_traces.add(str_trace)
        assert str_trace in expected_traces, "\n" + str_trace
    assert len(expected_traces) == len(actual_traces)


def debug_print_spaces(spaces: List[Schedule], trace_as_list: bool) -> None:
    for i, space in enumerate(spaces):
        print(f"##### Space {i}")
        print(space.mod.script())
        trace = Trace(space.trace.insts, {})
        trace = trace.simplified(remove_postproc=True)
        if trace_as_list:
            print(str(trace).strip().splitlines())
        else:
            print(trace)
