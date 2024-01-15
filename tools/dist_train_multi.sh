CONFIG=$1
NODES=$2

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NODES \
    --node_rank=$RANK  \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=1 \
    --master_port=$MASTER_PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
