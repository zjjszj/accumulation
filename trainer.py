from FCN.config import cfg
from FCN.data.build import make_data_loader
import torch.nn as nn
from FCN.model import build_model_optim
import time
import math
import torch.optim.lr_scheduler as lr_scheduler
import torch
from tqdm import tqdm
from FCN.layers import cross_entropy2d
from FCN.utils import label_accuracy_score
from FCN.engine.inference import Inference
import os
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'
best = wdir + 'best.pt'


def train(cfg, model_cfg='FCN/configs/vgg16-fcn32s.cfg'):
    epochs=cfg.SOLVER.MAX_EPOCHS
    start_epoch=0
    device = cfg.MODEL.DEVICE
    results_file=cfg.RESULT_FILE
    nc=cfg.MODEL.NUM_CLASSES        # number of classes

    # dataset
    train_loader=make_data_loader(cfg, is_train=True)
    val_loader=make_data_loader(cfg, is_train=False)

    # building model and optimizer also reuse.
    r=build_model_optim(cfg, model_cfg)
    model=r['model'].to(device=device)
    optimizer=r['optimizer']
    if cfg.MODEL.REFUSE.WEIGHT.strip():
        start_epoch=r['epoch']+1
        best_fitness=r['best_fitness']
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine  ## 越来越少
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = start_epoch - 1

    # inference object
    inference=Inference(cfg, model, val_loader, cross_entropy2d, device)

    # train
    t0=time.time()
    for epoch in range(start_epoch, epochs):
        eval_loss,  eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc= 0, 0, 0, 0, 0

        model.train()

        mloss=torch.zeros(1)
        nb=len(train_loader)        # number of batch.
        pbar = tqdm(enumerate(train_loader), total=nb)  # progress bar
        for i, (imgs, targets) in pbar:
            #imgs, targets=imgs.to(device=device), targets.to(device=device)
            imgs=imgs.cuda()
            targets=targets.cuda()
            # --multi scale--

            print('imgs.shape=====================', imgs.shape)
            outputs=model(imgs)
            # outputs=imgs.repeat(1,7,1,1).requires_grad_(True)
            loss=cross_entropy2d(outputs, targets)      # per sample
            print('loss===============', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # metric
            label_pred=outputs.max(dim=1)[1].cpu().numpy()
            label_true=targets.cpu().numpy()
            for lbp, lbt in zip(label_pred, label_true):
                acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, nc)
                eval_acc += acc
                eval_acc_cls += acc_cls
                eval_mean_iu += mean_iu
                eval_fwavacc += fwavacc

            mloss=(mloss*i+loss)/(i+1)   # mean loss per batch
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' + '%10.3g' * 2) % ('%g/%g' % (epoch, epochs - 1), mem, mloss)
            pbar.set_description(s)     # batch show

        scheduler.step()

        # test
        final_epoch=epoch+1==epochs
        if not opt.notest or final_epoch:
            results=inference.evaluate()

        # write result (train + val) accumulation
        with open(results_file, 'a') as f:
            f.write(s+'%5.5g'*5 % results + '\n')

        # tensorboard (train + val)
        train_results=[mloss, eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc]
        if tb_writer:
            tags=['train/loss', 'train/eval_acc', 'train/eval_acc_cls', 'train/eval_mean_iu', 'train/eval_fwavacc',
                  'val/loss', 'val/eval_acc', 'val/eval_acc_cls', 'val/eval_mean_iu', 'val/eval_fwavacc']
            for tag, l in zip(tags, train_results+list(results)):
                tb_writer.add_scalar(tag, l, epoch)

        # update acc
        best_fitness=list(results)[0] if list(results)[0]>best_fitness else 0.0

        # save model: save model best and last epoch.
        if best_fitness or final_epoch:
            with open(results_file, 'r') as f:
                chkpt = {
                    'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}
            if best_fitness:
                torch.save(chkpt, best)
            else:
                torch.save(chkpt, last)
            del chkpt

        # end epoch--------------------------------------------------
    # end training






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch FCN Training")
    parser.add_argument("--config_file", default="FCN/config/defaults.py", help="path to config file", type=str)
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    opt = parser.parse_args()

    # merge opt to cfg.
    cfg.merge_from_list(opt.opts)
    cfg.freeze()

    # tb_writer
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()

    train(cfg)















