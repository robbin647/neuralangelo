export GROUP='example'
export NAME='synthetic_chair'
echo "******* Running neuralangelo on $NAME ******"
export LOGDIR="/root/autodl-tmp/data/neuralangelo_log/$NAME"

torchrun --nproc_per_node=1 --master-port=29504 train.py \
    --logdir=$LOGDIR \
    --config=projects/neuralangelo/configs/custom/$NAME.yaml \
    --max_iter=20000 
    # --validation_iter=1000 \
    # --model.object.sdf.encoding.coarse2fine.step=200 \
    # --model.object.sdf.encoding.hashgrid.dict_size=19 \
    # --optim.sched.warm_up_end=200 \
    # --optim.sched.two_steps=[12000,16000]
    # --data.readjust.scale="$readjust_scale" \
