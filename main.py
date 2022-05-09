import pytorch_lightning as pl
from argparse import ArgumentParser
from models import cctv_trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
import torch
import pdb


def main():
    parser = ArgumentParser()

    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--checkpoint', default='', type=str, help='load trained weights')

    parser = cctv_trainer.CCTVPipeTrainer.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        save_last=True,
        verbose=True,
        monitor='val_mAP',
        every_n_epochs=1,
        mode='max',
        save_weights_only=True
    )
    trainer_obj = pl.Trainer.from_argparse_args(args, checkpoint_callback=checkpoint_callback)
    model = cctv_trainer.CCTVPipeTrainer(args)
    if args.checkpoint != '':
        checkpoint = torch.load(args.checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    trainer_obj.fit(model)


if __name__ == '__main__':
    main()
