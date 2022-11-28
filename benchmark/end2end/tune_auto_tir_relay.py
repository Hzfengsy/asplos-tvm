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
import numpy as np  # type: ignore

import tvm
from tvm import relay, runtime
from tvm import meta_schedule as ms
from tvm.contrib.graph_executor import GraphModule
from tvm.meta_schedule.testing.relay_workload import get_network
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.meta_schedule import extract_task_from_relay
from tvm.meta_schedule.tune import tune_extracted_tasks
from benchmark.utils import parse_args, load_config
from benchmark.end2end.utils import *

# We use fp16-16-32 intrinsic for e2e workloads
from tvm.meta_schedule.testing import tir_tensor_intrin_fp16

from benchmark.utils import *
from benchmark.end2end.utils import *

import logging
logging.basicConfig(level=logging.INFO)

WORKLOADS = ["resnet_50", "mobilenet_v2", "bert_large", "vit"]
ARGS = parse_args(WORKLOADS)


def f_measurement(rt_mod: runtime.Module, device: runtime.ndarray.Device, input_data):
    mod = GraphModule(rt_mod["default"](device))
    for input_name, input_value in input_data.items():
        mod.set_input(input_name, input_value)
    evaluator = mod.module.time_evaluator(
        "run",
        device,
        min_repeat_ms=500,
        repeat=3,
    )
    print(evaluator())


def tune(workload, input_shape):
    os.makedirs("benchmark/caches/relay")
    relay_mod, params, (input_name, input_shape, input_dtype) = get_network(
        workload,
        input_shape,
        cache_dir="benchmark/caches/relay",
    )

    relay_mod = convert_conv2d_layout(relay_mod, {"nn.conv2d": ["NHWC", "OHWI"]})
    relay_mod = relay.transform.ToMixedPrecision("float16")(relay_mod)
    relay_mod = rewrite_reshape_gelu(relay_mod)
    tasks = extract_task_from_relay(relay_mod, target=ARGS.target, params=params)

    # run tuning tasks
    print("Tuning...")
    memhammer_tasks = []
    other_tasks = []
    for tsk in tasks:
        if should_use_memhammer(tsk):
            print(tsk.task_name, "memhammer")
            memhammer_tasks.append(tsk)
        else:
            print(tsk.task_name, "non-memhammer")
            other_tasks.append(tsk)

    search_config = get_search_config(ARGS.num_trials, 1000)
    database = tune_extracted_tasks(
        memhammer_tasks,
        config=search_config,
        sch_rules=sch_rules_tensor_core,
        postprocs=postprocs_tensor_core,
        work_dir=f"{ARGS.work_dir}/TIR/{workload}-{input_shape}",
        runner=ARGS.runner,
    )

    database = tune_extracted_tasks(
        other_tasks,
        config=search_config,
        # use default CUDA rules
        work_dir=f"{ARGS.work_dir}/TIR/{workload}-{input_shape}",
        database=database,
        runner=ARGS.runner,
    )

    with ms.ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_meta_schedule": True, "tir.predicate_opt": True},
        ):
            mod = tvm.relay.build(relay_mod, target=ARGS.target, params=params)

    if input_dtype.startswith("float"):
        input_data = {
            input_name: np.random.uniform(size=input_shape).astype(input_dtype)
        }
    else:
        input_data = {
            input_name: np.random.randint(
                low=0, high=10000, size=input_shape, dtype=input_dtype
            )
        }

    if ARGS.use_rpc:
        run_module_via_rpc(
            rpc_config=ARGS.runner.rpc_config,
            lib=mod,
            dev_type=ARGS.target.kind.name,
            args=input_data,
            continuation=f_measurement,
        )
    else:
        dev = tvm.device(ARGS.target.kind.name)
        input_data = {
            key: tvm.runtime.ndarray.array(value, dev)
            for key, value in input_data.items()
        }
        f_measurement(mod, dev, input_data)


if __name__ == "__main__":
    configs = load_config()
    for workload in ARGS.workload:
        shape = configs[workload]
        for batch in ARGS.batch_size:
            shape = [batch] + shape[1:]
            tune(workload, shape)
