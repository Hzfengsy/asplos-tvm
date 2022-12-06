import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tvm
from torchvision.models import quantization as qmodels
from tvm import IRModule
from tvm import meta_schedule as ms
from tvm import relax, relay, runtime
from tvm.ir import transform
from tvm.meta_schedule.testing.custom_builder_runner import run_module_via_rpc
from tvm.relax.testing import relay_translator
from tvm.runtime import load_param_dict, save_param_dict
from tvm.target import Target

from benchmark import utils

ARGS = utils.parse_args(["resnet_18", "resnet_50", "mobilenet_v2"])


def _load_cache(cache_dir: Optional[str], filename: str) -> Optional[List[Any]]:
    if cache_dir is None:
        return None
    path = os.path.join(os.path.expanduser(cache_dir), filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as i_f:
        return pickle.load(i_f)


def _save_cache(cache_dir: Optional[str], filename: str, objects: List[Any]) -> None:
    if cache_dir is None:
        return
    path = os.path.join(os.path.expanduser(cache_dir), filename)
    with open(path, "wb") as o_f:
        pickle.dump(objects, o_f)


def _get_model(name, input_shape):
    if name == "resnet_18":
        model = qmodels.resnet18()
    elif name == "resnet_50":
        model = qmodels.resnet50()
    elif name == "mobilenet_v2":
        model = qmodels.mobilenet_v2()

    input_data = torch.rand(input_shape)

    model.eval()
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("qnnpack")
    model = torch.quantization.prepare(model)
    model(input_data)
    model = torch.quantization.convert(model)

    script_module = torch.jit.trace(model, input_data).eval()
    mod, params = relay.frontend.from_pytorch(script_module, [("data", input_shape)])
    params_bytearray: bytearray = save_param_dict(params)
    return mod, params_bytearray


def get_model(name, input_shape, cache_dir: str):
    os.makedirs(cache_dir, exist_ok=True)
    filename = f'relay-qnn-{name}-{",".join(str(i) for i in input_shape)}.json'
    cached = _load_cache(cache_dir, filename)
    if cached is None:
        mod, params_bytearray = _get_model(name, input_shape)
        cached = [mod, params_bytearray]
        _save_cache(cache_dir, filename, cached)
    mod, params_bytearray = cached
    params = load_param_dict(params_bytearray)
    return mod, params


def apply_opt_before_tuning(
    relay_mod: IRModule, params: Dict[str, runtime.NDArray], target: Target
):
    with transform.PassContext(opt_level=3, config={"relay.backend.use_meta_schedule": True}):
        main_func = relay_mod["main"]
        bind_main_func = relay.build_module.bind_params_by_name(main_func, params)
        relay_mod = IRModule.from_expr(bind_main_func)
        # translator contains prepass for relay.
        relax_mod = relay_translator.from_relay(relay_mod["main"], target=target)
        relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
        relax_mod = relax.transform.FuseOps()(relax_mod)
        relax_mod = relax.transform.FuseTIR()(relax_mod)
    return relax_mod


def tune(workload, input_shape):
    relay_mod, params = get_model(workload, input_shape, "benchmark/caches/relay")

    # translate the ResNet model from Relay to Relax
    relax_mod = apply_opt_before_tuning(relay_mod, params, target=ARGS.target)
    assert isinstance(relax_mod, IRModule)
    executable = ms.tune_relax(
        mod=relax_mod,
        target=ARGS.target,
        config=utils.get_search_config(ARGS.num_trials, ARGS.num_trials),
        work_dir=f"{ARGS.work_dir}/TVM/{workload}",
        runner=ARGS.runner,  # type: ignore
    )
    inputs = {"data": tvm.nd.array(np.random.uniform(0, 255, input_shape).astype("float32"))}
    run_module_via_rpc(
        rpc_config=ARGS.runner.rpc_config,
        lib=executable.mod,
        dev_type=ARGS.target.kind.name,
        args=inputs,
        continuation=utils.f_relax_measurement,
    )


if __name__ == "__main__":
    configs = utils.load_config()
    for workload in ARGS.workload:
        shape = configs[workload]
        tune(workload, shape)
