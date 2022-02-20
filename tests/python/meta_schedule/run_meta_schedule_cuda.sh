set -euxo pipefail

RPC_HOST="192.168.6.66"
RPC_PORT="4445"
RPC_KEY="jetson-agx-xavier"
TARGET="nvidia/jetson-agx-xavier"
LOG_DIR=$HOME/logs/ms-cuda/
NUM_TRIALS=2000

mkdir -p $LOG_DIR

run () {
    name=$1
    work_dir=$LOG_DIR/$name/
    mkdir -p $work_dir
    echo "Running workload $name"
    python tests/python/meta_schedule/test_meta_schedule.py \
        --workload "$name"                  \
        --target "$TARGET"                  \
        --work-dir "$work_dir"              \
        --rpc-host "$RPC_HOST"              \
        --rpc-port "$RPC_PORT"              \
        --rpc-key "$RPC_KEY"                \
        --num-trials $NUM_TRIALS            \
        2>&1 | tee "$work_dir/$name.log"
}

run C1D
run C2D
run CAP
run DEP
run DIL
run GMM
run GRP
run T2D
run C2d-BN-RELU
run TBG

run C3D
run NRM
run SFM
