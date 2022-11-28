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
import numpy as np
import os

import tvm
from tvm import meta_schedule as ms
from tvm.relay.transform import ToMixedPrecision
from tvm.meta_schedule.testing import relay_workload
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc

from benchmark.utils import parse_args

_ = 1
CONFIG = {
    "resnet_50": [_, 3, 224, 224],
    "mobilenet_v2": [_, 3, 224, 224],
    "bert_large": [_, 256],
    "vit": [_, 3, 224, 224],
}


ARGS = parse_args(CONFIG.keys())
CACHE_DIR = "benchmark/caches"
if not os.path.exists:
    os.mkdir(CACHE_DIR)


def tune(workload, input_shape):
    mod, params, (input_name, input_shape, input_dtype) = relay_workload.get_network(
        workload,
        input_shape,
        cache_dir="benchmark/caches/relay",
    )
    mod = ToMixedPrecision("float16")(mod)
    input_info = {input_name: input_shape}
    input_data = {}
    print(f"Workload: {workload}")
    for input_name, input_shape in input_info.items():
        print(f"  input_name: {input_name}")
        print(f"  input_shape: {input_shape}")
        print(f"  input_dtype: {input_dtype}")
    lib = ms.tune_relay(
        mod=mod,
        target=ARGS.target,
        config=ms.TuneConfig(
            strategy="evolutionary",
            num_trials_per_iter=64,
            max_trials_per_task=2000,
            max_trials_global=ARGS.num_trials,
        ),
        runner=ARGS.runner,
        work_dir=f"{ARGS.work_dir}/TVM/{workload}-{input_shape}",
        params=params,
    )
    print("Tuning Time:")
    graph, rt_mod, params = lib.graph_json, lib.lib, lib.params
    for input_name, input_shape in input_info.items():
        if input_dtype.startswith("float"):
            input_data[input_name] = np.random.uniform(size=input_shape).astype(
                input_dtype
            )
        else:
            input_data[input_name] = np.random.randint(
                low=0, high=10000, size=input_shape, dtype=input_dtype
            )

    def f_measurement(rt_mod, dev, input_data):
        # pylint: disable=import-outside-toplevel
        from tvm.contrib.graph_executor import GraphModule

        # pylint: enable=import-outside-toplevel

        mod = GraphModule(rt_mod["default"](dev))
        for input_name, input_value in input_data.items():
            mod.set_input(input_name, input_value)
        evaluator = mod.module.time_evaluator(
            "run",
            dev,
            min_repeat_ms=500,
            repeat=3,
        )
        print(evaluator())

    if ARGS.use_rpc:
        run_module_via_rpc(
            rpc_config=ARGS.runner.rpc_config,
            lib=lib,
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
        f_measurement(lib, dev, input_data)


if __name__ == "__main__":
    for workload in ARGS.workload:
        shape = CONFIG[workload]
        for batch in ARGS.batch_size:
            shape = [batch] + shape[1:]
            tune(workload, shape)
