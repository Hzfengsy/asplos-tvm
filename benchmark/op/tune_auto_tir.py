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

import tvm
from tvm import meta_schedule as ms
from workloads_fp16 import create_te_workload_f16
from benchmark.utils import *

WORKLOADS = [
    "C1D",
    "C2D",
    "C3D",
    "DIL",
    "DEP",
    "GRP",
    "T2D",
    "CBR",
    "GMM-1024",
    "GMM-4096",
]
ARGS = parse_args(WORKLOADS, default_trials=1000)

if ARGS.out_dtype == "float16":
    from tvm.meta_schedule.testing import tir_tensor_intrin_fp16
elif ARGS.out_dtype == "float32":
    from tvm.meta_schedule.testing import tir_tensor_intrin
else:
    raise Exception("Unsupported dtype")


def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(config={"tir.predicate_opt": True}):
        return tvm_build(mod, target=target)


def tune(workload, batch_size=1):
    mod = create_te_workload_f16(
        workload, batch_size=batch_size, out_dtype=ARGS.out_dtype
    )
    sch = ms.tune_tir(
        mod=mod,
        target=ARGS.target,
        config=get_search_config(ARGS.num_trials, ARGS.num_trials),
        work_dir=f"{ARGS.work_dir}/TIR/{workload}-{batch_size}/{ARGS.out_dtype}",
        builder=ms.builder.LocalBuilder(f_build=cuda_build),
        runner=ARGS.runner,  # type: ignore
        sch_rules=sch_rules_tensor_core,
        postprocs=postprocs_tensor_core,
    )

    if sch is None:
        print("No valid schedule found!")
        exit()

    print(sch.mod.script())
    print(sch.trace)


if __name__ == "__main__":
    for workload in ARGS.workload:
        for batch in ARGS.batch_size:
            tune(workload, batch)
