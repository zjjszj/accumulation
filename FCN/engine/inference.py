from tqdm import tqdm
import torch
from FCN.utils import label_accuracy_score

class Inference:
    def __init__(self, cfg, model, loader, criterion_fn, device):
        self.cfg=cfg
        self.model=model
        self.loader=loader
        self.criterion=criterion_fn
        self.device=device

    def evaluate(self):
        """return mloss and metrics

        """
        self.model.eval()

        mloss=torch.zeros(1)
        eval_loss,  eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc= 0, 0, 0, 0, 0
        with torch.no_grad():
            pbar=tqdm(enumerate(self.loader), total=len(self.loader))
            for i, data in pbar:
                imgs, label=data
                imgs, label=imgs.to(self.device), label.to(self.device)
                out=self.model(imgs)
                loss=self.criterion(out, label)
                mloss=(mloss*i+loss)/(i+1)

                label_pred = out.max(dim=1)[1].data.cpu().numpy()
                label_true = label.data.cpu().numpy()
                for lbt, lbp in zip(label_true, label_pred):
                    acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbt, lbp, self.cfg.MODEL.NUM_CLASSES )
                    eval_acc += acc
                    eval_acc_cls += acc_cls
                    eval_mean_iu += mean_iu
                    eval_fwavacc += fwavacc

                s=('loss=%.5g\teval_acc=%5.g\teval_acc_cls=%5.g\teval_mean_iu=%5.g\teval_fwavacc=%5.g')%(loss, eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc)
                pbar.set_description(s)
                # print(s)
        return mloss, eval_acc, eval_acc_cls, eval_mean_iu, eval_fwavacc

