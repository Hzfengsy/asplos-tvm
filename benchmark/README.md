# Experiment Scripts for the Paper "TensorIR: An Abstraction for Automatic Tensorized Program Optimization"

To run the benchmarks of TensorIR, we've prepared a series of scripts to reproduce the results we showed in the paper.

## Prepare

Before running the scripts, let's first set some environment variables:

```bash
# Set the target, please remember to modify the core number
export TVM_TARGET="llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+neon,+v8.2a,+dotprod -num-cores X"

# Set the RPC settings
export TVM_RPC_HOST="172.16.2.241"
export TVM_RPC_PORT=4445
export TVM_RPC_KEY="ARM"
```

## Single Op Benchmark

Here we only provide the simple ways to run the scripts. If you'd like to customize the settings, you can see the all possible settings by running:

```bash
python benchmark/xx/xx_script.py -h
```

**NOTE: All scripts need to be run under the root directory of the TVM project.**

### Cutlass

```bash
python benchmark/op/run_acl.py -w GMM-1024
```

### TVM

```bash
python benchmark/op/tune_tvm.py -w GMM-1024
```

### TensorIR

```bash
python benchmark/op/tune_auto_tir.py.py -w GMM-1024
```

## End-to-End Benchmark

Here we only provide the simple ways to run the scripts. If you'd like to customize the settings, you can see the all possible settings by running:

```bash
python benchmark/xx/xx_script.py -h
```

### TVM

```bash
python benchmark/end2end/tune_qnn_tvm_relax.py -w resnet_50
```

### TensorIR

```bash
python benchmark/end2end/tune_qnn_auto_tir_relax.py -w resnet_50
```

### PyTorch

```bash
python benchmark/end2end/pytorch.py -w resnet_50
```
