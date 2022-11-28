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
from typing import Optional

from tvm import tir
from tvm import meta_schedule as ms
from workloads_fp16 import create_te_workload_f16

from benchmark.utils import parse_args, get_search_config

WORKLOADS = [
    "C1D",
    "C2D",
    "C3D",
    "DIL",
    "GRP",
    "DEP",
    "T2D",
    "CBR",
    "GMM-1024",
    "GMM-4096",
]

ARGS = parse_args(WORKLOADS, 2000)


def tune(workload, batch_size):
    sch: Optional[tir.Schedule] = ms.tune_tir(
        mod=create_te_workload_f16(workload, batch_size, ARGS.out_dtype),
        target=ARGS.target,
        config=get_search_config(ARGS.num_trials, ARGS.num_trials),
        runner=ARGS.runner,
        work_dir=f"{ARGS.work_dir}/TVM/{workload}-{batch_size}/{ARGS.out_dtype}/",
    )
    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    for workload in ARGS.workload:
        for batch_size in ARGS.batch_size:
            tune(workload, batch_size)
