set -euxo pipefail

RPC_HOST="192.168.6.66"
RPC_PORT="4445"
RPC_KEY="raspi4b-aarch64"
TARGET="raspberry-pi/4b-64"
NUM_TRIALS=800
LOG_DIR=$HOME/logs/ansor-cpu/

mkdir -p $LOG_DIR

run () {
    name=$1
    echo "Running workload $name"
    python tests/python/meta_schedule/test_ansor_cpu.py \
        --workload "$name"                  \
        --target "$TARGET"                  \
        --rpc-host "$RPC_HOST"              \
        --rpc-port "$RPC_PORT"              \
        --rpc-key "$RPC_KEY"                \
        --num-trials "$NUM_TRIALS"          \
        --log-dir $LOG_DIR                  \
        2>&1 | tee "$LOG_DIR/$name.log"
}

# Single op
run C1D
run C2D
run C3D
run CAP
run DEP
run DIL
run GMM
run GRP
run NRM
run T2D
# Subgraph
run C2d-BN-RELU
run TBG

