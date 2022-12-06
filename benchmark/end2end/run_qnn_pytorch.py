import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torchvision.models import quantization as qmodels

from benchmark.utils import load_config, parse_args

ARGS = parse_args(["resnet_18", "resnet_50", "mobilenet_v2"])


def profile_model(name, input_shape, warmup=50, repeat=100):
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
    # Dummy calibration
    model(input_data)
    model = torch.quantization.convert(model)

    activities = [ProfilerActivity.CPU]

    for _ in range(warmup):
        model(input_data)
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("model_inference"):
            for _ in range(repeat):
                model(input_data)

    print(prof.key_averages().table(sort_by=f"cpu_time_total", row_limit=10))


if __name__ == "__main__":
    torch.backends.quantized.engine = "qnnpack"
    configs = load_config()
    for workload in ARGS.workload:
        shape = configs[workload]
        profile_model(workload, shape)
