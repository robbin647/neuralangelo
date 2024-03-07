GROUP='example'
NAME='dtu_scan24'
readjust_scale='0.7'

torchrun --nproc_per_node=1 train.py \
    --logdir="logs/$GROUP/$NAME" \
    --config=projects/neuralangelo/configs/custom/dtu_scan24.yaml \
    --data.readjust.scale="$readjust_scale" \
    --max_iter=20000 \
    --validation_iter=99999999 \
    --model.object.sdf.encoding.coarse2fine.step=200 \
    --model.object.sdf.encoding.hashgrid.dict_size=19 \
    --optim.sched.warm_up_end=200 \
    --optim.sched.two_steps=[12000,16000]
