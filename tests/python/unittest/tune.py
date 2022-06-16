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
# pylint: disable=missing-docstring
from typing import Tuple

import logging
import tempfile

import tvm
import pytest
import tvm.topi.testing
from tvm import te, topi, tir, meta_schedule
from tvm.meta_schedule import tune_tir
from tvm.meta_schedule.tune import TuneConfig
from tvm.meta_schedule.tune_context import TuneContext
from tvm.meta_schedule import schedule_rule, postproc
from tvm.meta_schedule.space_generator import PostOrderApply
from tvm.ir import IRModule
from tvm.script import tir as T
from tvm.target.target import Target
from tvm.te.operation import create_prim_func
from tvm.tir import Schedule
from tvm.meta_schedule.testing import te_workload, tir_tensor_intrin
from tvm.meta_schedule.testing.te_workload_tc import create_te_workload_f16

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)

WORKLOAD = "GMM"
prim_func = create_te_workload_f16(WORKLOAD, 0)
mod = IRModule({"main": prim_func})

print(mod.script())

target = Target("nvidia/geforce-rtx-3070")
config = TuneConfig(
    num_trials_per_iter=64,
    max_trials_per_task=1000,
    max_trials_global=2000,
    search_strategy_config={
        "population_size": 2048,
        "init_measured_ratio": 0.2,
        "init_min_unmeasured": 50,
        "genetic_num_iters": 3,
        "genetic_mutate_prob": 0.85,
        "genetic_max_fail_count": 10,
        "eps_greedy": 0.05,
    })


class DefaultTensorCore:
    @staticmethod
    def _sch_rules():
        from tvm.meta_schedule import (  # pylint: disable=import-outside-toplevel
            schedule_rule as M,
        )

        return [
            M.MultiLevelTiling(
                structure="SSSRRSRS",
                tile_binds=["blockIdx.x", "blockIdx.y", "threadIdx.y"],
                use_tensor_core=True,
                max_innermost_factor=4,
                vector_load_lens=[1, 2, 3, 4],
                reuse_read=schedule_rule.ReuseType(
                    req="must",
                    levels=[4],
                    scope="shared",
                ),
                reuse_write=schedule_rule.ReuseType(
                    req="no",
                    levels=[3],
                    scope="shared",
                ),
            ),
            M.AutoInline(
                into_producer=True,
                into_consumer=True,
                inline_const_tensor=True,
                disallow_if_then_else=False,
                require_injective=False,
                require_ordered=False,
                disallow_op=None,
            ),
            M.CrossThreadReduction(thread_extents=[4, 8, 16, 32, 64, 128, 256, 512]),
            M.ParallelizeVectorizeUnroll(
                max_jobs_per_core=-1,  # disable parallelize
                max_vectorize_extent=-1,  # disable vectorize
                unroll_max_steps=[0, 16, 64, 512, 1024],
                unroll_explicit=True,
            ),
        ]

    @staticmethod
    def _postproc():
        from tvm.meta_schedule import (  # pylint: disable=import-outside-toplevel
            postproc as M,
        )

        return [
            M.RewriteCooperativeFetch(),
            M.RewriteParallelVectorizeUnroll(),
            M.RewriteReductionBlock(),
            M.RewriteTensorCore(),
            M.VerifyGPUCode(),
        ]

    @staticmethod
    def _mutator_probs():
        from tvm.meta_schedule import mutator as M

        return {
            M.MutateTileSize(): 0.9,
            M.MutateUnroll(): 0.1,
        }


with tempfile.TemporaryDirectory() as work_dir:
    sch: Schedule = tune_tir(
        mod=mod,
        target=target,
        config=config,
        work_dir=work_dir,
        space=PostOrderApply(),
        sch_rules=DefaultTensorCore._sch_rules,
        postprocs=DefaultTensorCore._postproc,
        mutator_probs=DefaultTensorCore._mutator_probs,
        num_threads=None,
    )
    if sch is None:
        print("No valid schedule found!")
        exit()

    print(sch.mod.script())
    print("".join(sch.trace.as_python()))

