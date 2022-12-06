import numpy as np
import tvm
from infrastructure import Device, build_and_run
from benchmark import utils


WORKLOADS = ["C2D-1", "C2D-2", "GMM-1024", "GMM-2048", "GMM-4096"]
ARGS = utils.parse_args(WORKLOADS, default_trials=1000)
configs = utils.load_config()

Device.host = ARGS.rpc_host
Device.port = ARGS.rpc_port
Device.device_key = ARGS.rpc_key
Device.connection_type = "tracker"
Device.target = ARGS.target
device = Device()

for workload in ARGS.workload:
    config = configs.get(workload)
    assert config is not None
    print(f"Running {workload}")
    if "GMM" in workload:
        shape, weight_shape, units = config
        inputs = {"data": tvm.nd.array(np.random.uniform(-128, 127, shape).astype("int8"))}
        func, params = utils.get_qnn_gemm_model(shape, weight_shape, units)
    elif "C2D" in workload:
        kernel_h, kernel_w, pad, stride, dilation, out_channels, shape = config
        inputs = {"data": tvm.nd.array(np.random.uniform(0, 255, shape).astype("uint8"))}
        func, params = utils.get_qnn_conv2d_model(
            shape,
            kernel_h,
            kernel_w,
            pad,
            stride,
            dilation,
            out_channels,
        )

    build_and_run(func, inputs, 1, params, device, enable_acl=True, tvm_ops=0)[0]
