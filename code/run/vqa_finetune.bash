# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=snap/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src:./src/lxrt/SlowFast \
    python3 src/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 24 --optim bert --lr 1e-4 --epochs 30 \
    --tqdm --output $output ${@:3}
