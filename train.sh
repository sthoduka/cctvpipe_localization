#!/bin/sh
python main.py \
  --video_root=video_track \
  --batch_size=512 \
  --crop_size=300 \
  --resize_size=224 \
  --training_type=single_img \
  --group=5 \
  --default_root_dir=logs/ \
  --learning_rate=1.0 \
  --max_epochs=40 \
  --gpus=1 \
#  --checkpoint=path_to_previous_checkpoint_if_applicable.ckpt

