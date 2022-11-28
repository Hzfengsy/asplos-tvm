# Experiment Scripts for the Paper "TensorIR: An Abstraction for Automatic Tensorized Program Optimization"

To run the benchmarks of TensorIR, we've prepared a series of scripts to reproduce the results we showed in the paper.

## Prepare

Before running the scripts, let's first set some environment variables:

```bash
# Set the target you want
export TVM_TARGET="the_target_you_want_to_run"
# The target list can be found in `src/target/tag.cc`. e.g.
# export TVM_TARGET="nvidia/geforce-rtx-3080"

# Set the cutlass home path if you want to run the cutlass benchmarks
export CUTLASS_HOME="path/to/cutlass"
# e.g. export CUTLASS_HOME="/home/xxx/cutlass"

# Set the RPC settings if you want to use the RFC runner, e.g.
# export TVM_RPC_HOST="172.16.2.241"
# export TVM_RPC_PORT=4445
# export TVM_RPC_KEY="rtx-3080"
```

## Single Op Benchmark

Here we only provide the simple ways to run the scripts. If you'd like to customize the settings, you can see the all possible settings by running:

```bash
python benchmark/xx/xx_script.py -h
```

**NOTE: All scripts need to be run under the root directory of the TVM project.**

### Cutlass

```bash
python benchmark/op/run_cutlass.py -w GMM-1024
```

### TVM

```bash
python benchmark/op/tune_tvm.py -w GMM-1024 -bs 1 -n 1000
```

### TensorIR

```bash
python benchmark/op/tune_auto_tir.py.py -w GMM-1024 -bs 1 -n 1000
```

## End-to-End Benchmark

Here we only provide the simple ways to run the scripts. If you'd like to customize the settings, you can see the all possible settings by running:

```bash
python benchmark/xx/xx_script.py -h
```

### TVM

```bash
python benchmark/end2end/tune_tvm.py -bs 1 -w resnet_50 -n 20000
```

### TensorIR

We've integrated the TensorIR tuner into the both relay and relax. There should be no performance difference between them.

```bash
# Relay-based:
python benchmark/end2end/tune_auto_tir_relay.py -w resnet_50 -bs 1 -n 10000
# Relax-based:
python benchmark/end2end/tune_auto_tir_relax.py -w resnet_50 -bs 1 -n 10000
```

## PyTorch and PyTorch-TensorRT

### PyTorch

```bash
python benchmark/pytorch.py -w GMM-1024 -n 1 16
```

### PyTorch-TensorRT

```bash
python benchmark/pytorch-tensorrt.py -w GMM-1024 -n 1 16
```

## AMOS

Please see AMOS official repo: https://github.com/pku-liang/AMOS