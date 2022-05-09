import pytorch_lightning as pl
from argparse import ArgumentParser
from models import cctv_trainer
import pdb
import torch
import numpy as np
import os
import json
import torchnet.meter as meter


def main():
    parser = ArgumentParser()

    parser.add_argument('--video_root', default='', type=str, help='Root path of input videos')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('--n_threads', default=4, type=int, help='Number of threads for multi-thread loading')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint file')
    parser.add_argument('--data_type', default='test', type=str, help='test or val')

    parser = cctv_trainer.CCTVPipeTrainer.add_model_specific_args(parser)

    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    device = 'cuda:0'

    model = cctv_trainer.CCTVPipeTrainer.load_from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.hparams.batch_size = args.batch_size
    model.hparams.n_threads = args.n_threads


    model.eval()

    data_type = args.data_type
    if data_type == 'val':
        test_loader = model.val_dataloader()
    else:
        test_loader = model.test_dataloader()
    total = 0

    output = {}

    mapmeter = meter.mAPMeter()
    mapmeter_norm = meter.mAPMeter()

    with torch.no_grad():
        idx_to_label = test_loader.dataset.idx_to_label
        all_results = {}
        all_results["results"] = {}
        for batch in test_loader:
            img, label, names, moments = batch
            img = img.to(device)
            out = model(img)
            out = torch.sigmoid(out)
            for idx, res in enumerate(out):
                nm = names[idx]
                moment = moments[idx].detach().cpu().item()
                res2 = res[:-1]
                for clsidx, score in enumerate(res2):
                    label = idx_to_label[clsidx]
                    res_obj = {'score': "%.3f" % score , 'moment': "%.3f" % moment, 'label': label, 'nominal_score': '%.3f' % res[-1]}
                    if nm not in all_results["results"].keys():
                        all_results["results"][nm] = []
                    all_results["results"][nm].append(res_obj)
            total += 1
            print("%d / %d" % (total * args.batch_size, len(test_loader.dataset)), flush=True)
    with open('%s_result_%s.json' % (args.data_type, os.path.basename(args.checkpoint)), 'w') as fp:
        json.dump(all_results, fp)

if __name__ == '__main__':
    main()
