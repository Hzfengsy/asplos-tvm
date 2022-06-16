RPC_HOST="172.16.2.241"
RPC_PORT=4445

RPC_KEY="rtx-3080"
TARGET="nvidia/geforce-rtx-3080"
LOG_DIR=logs/relax-cuda/
CMD="python3 e2e_bert.py"

mkdir -p $LOG_DIR

run () {
  workload="$1"
  input_shape="$2"
  num_trials="$3"
  tune_model="$4"
  WORK_DIR=$LOG_DIR/$workload/
  mkdir -p $WORK_DIR

  $CMD                                \
    --workload "$workload"            \
    --input-shape "$input_shape"      \
    --target "$TARGET"                \
    --num-trials $num_trials          \
    --rpc-host $RPC_HOST              \
    --rpc-port $RPC_PORT              \
    --rpc-key $RPC_KEY                \
    --work-dir $WORK_DIR              \
    --cache-dir $HOME/cache-workloads \
    2>&1 | tee "$WORK_DIR/$workload.log"
}

# set trials=0 to disable tuning
run "bert"     "[8,16]"   0