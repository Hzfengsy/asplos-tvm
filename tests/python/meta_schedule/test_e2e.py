from typing import List, Tuple

from tvm.meta_schedule.testing.e2e import extract, get_network
from tvm.target import Target

MODEL_CACHE_DIR = "~/dataset/relay-models"
TASK_CACHE_DIR = "~/dataset/tasks-{target_kind}"


def _build_dataset() -> List[Tuple[str, List[int]]]:
    network_keys = []
    for name in [
        "resnet_18",
        "resnet_50",
        "mobilenet_v2",
        "mobilenet_v3",
        "wide_resnet_50",
        "resnext_50",
        "densenet_121",
    ]:
        for batch_size in [1, 4, 8]:
            for image_size in [224, 240, 256]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # inception-v3
    for name in ["inception_v3"]:
        for batch_size in [1, 2, 4]:
            for image_size in [299]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # resnet3d
    for name in ["resnet3d_18"]:
        for batch_size in [1, 2, 4]:
            for image_size in [112, 128, 144]:
                network_keys.append((name, [batch_size, 3, image_size, image_size, 16]))
    # bert
    for name in ["bert_tiny", "bert_base", "bert_medium", "bert_large"]:
        for batch_size in [1, 2, 4]:
            for seq_length in [64, 128, 256]:
                network_keys.append((name, [batch_size, seq_length]))
    # dcgan
    for name in ["dcgan"]:
        for batch_size in [1, 4, 8]:
            for image_size in [64]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))

    return network_keys


def test_import():
    network_keys = _build_dataset()
    for i, (name, input_shape) in enumerate(network_keys, 1):
        print(f"[{i} / {len(network_keys)}] Import {name}, input_shape = {input_shape}")
        get_network(name, input_shape, cache_dir=MODEL_CACHE_DIR)


def test_extract():
    network_keys = _build_dataset()
    for target_kind in ["llvm", "cuda"]:
        for i, (name, input_shape) in enumerate(network_keys, 1):
            print(
                f"[{i} / {len(network_keys)}] Extract {name} @ {target_kind}, input_shape = {input_shape}"
            )
            if name == "resnext_50" and target_kind == "cuda":
                continue
            mod, params, _ = get_network(name, input_shape, cache_dir=MODEL_CACHE_DIR)
            filename = f'{name}-{",".join(str(i) for i in input_shape)}-{target_kind}.json'
            extracted_tasks = extract(
                filename=filename,
                mod=mod,
                target=Target(target_kind),
                params=params,
                cache_dir=TASK_CACHE_DIR.format(target_kind=target_kind),
            )
            print(f"{len(extracted_tasks)} task(s) extracted")


if __name__ == "__main__":
    test_import()
    test_extract()
