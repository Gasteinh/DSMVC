import argparse
from dataloader import load_data
from torch.utils.data import DataLoader
from loss import Loss
from evaluation import evaluate
from model import DSMVC
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='caltech_5m')
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--epochs", default=120)
parser.add_argument("--view", type=int, default=2)
args = parser.parse_args()
if args.dataset == "mnist_mv":
    args.feature_dim = 288


def train_safe_epoch(epoch, view, model, data_loader, criterion, optimizer, records, device):
    tot_loss = 0.
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, _, output, hidden = model(xs)
        loss, _ = criterion.forward_cluster(hidden, output)
        loss.backward()
        optimizer.step(epoch - 1 + batch_idx / len(data_loader))
        tot_loss += loss.item()
    if epoch == 120:
        records['safe'].append(tot_loss/len(data_loader))


class Optimizer:
    def __init__(self, params):
        """
        Wrapper class for optimizers

        :param cfg: Optimizer config
        :type cfg: config.defaults.Optimizer
        :param params: Parameters to associate with the optimizer
        :type params:
        """
        self.clip_norm = 5.0
        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.5
        self.params = params
        self._opt = torch.optim.Adam(params, lr=1e-3)
        if self.scheduler_step_size is not None:
            assert self.scheduler_gamma is not None
            self._sch = torch.optim.lr_scheduler.StepLR(self._opt, step_size=self.scheduler_step_size,
                                                     gamma=self.scheduler_gamma)
        else:
            self._sch = None

    def zero_grad(self):
        return self._opt.zero_grad()

    def step(self, epoch):
        if self._sch is not None:
            # Only step the scheduler at integer epochs, and don't step on the first epoch.
            if epoch.is_integer() and epoch > 0:
                self._sch.step()

        if self.clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_norm)

        out = self._opt.step()
        return out


def valid(model, device, dataset, total_view):
    loader = DataLoader(
        dataset,
        batch_size=100,
        shuffle=False,
    )
    model.eval()
    pred_vector = []
    labels_vector = []

    for _, (xs, y, _) in enumerate(loader):
        for v in range(total_view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            _, _, output, _ = model(xs)
            pred_vector.extend(output.detach().cpu().numpy())
        labels_vector.extend(y.numpy())

    labels = np.array(labels_vector).reshape(len(labels_vector))
    pred_vec = np.argmax(np.array(pred_vector), axis=1)
    nmi, ari, acc, pur = evaluate(labels, pred_vec)
    return [acc, nmi, ari, pur]


def main():
    # prepare data and initial hyper-parameters
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_train_epoch = args.epochs
    view_num = args.view
    T = 20
    records = {"safe": [[], [], [], []]}
    loss_record = {"safe": []}

    for t in range(T):
        print('Iter:{}'.format(t))

        # initial safe MVC model
        safe_model = DSMVC(view_num-1, view_num, dims, args.feature_dim, class_num)
        safe_model = safe_model.to(device)
        criterion_safe = Loss(args.batch_size, class_num, device)

        for epoch in range(safe_train_epoch):
            if (epoch//20) % 2 == 0:
                for p in safe_model.gate.parameters():
                    p.requires_grad = False
                for p in safe_model.old_model.parameters():
                    p.requires_grad = True
                for p in safe_model.new_model.parameters():
                    p.requires_grad = True
                for p in safe_model.single.parameters():
                    p.requires_grad = True
                for p in safe_model.cluster_module.parameters():
                    p.requires_grad = True
                optimizer_theta = Optimizer(filter(lambda p: p.requires_grad, safe_model.parameters()))
                train_safe_epoch(epoch+1, view, safe_model, data_loader, criterion_safe, optimizer_theta, loss_record, device)
            else:
                for p in safe_model.gate.parameters():
                    p.requires_grad = True
                for p in safe_model.old_model.parameters():
                    p.requires_grad = False
                for p in safe_model.new_model.parameters():
                    p.requires_grad = False
                for p in safe_model.single.parameters():
                    p.requires_grad = False
                for p in safe_model.cluster_module.parameters():
                    p.requires_grad = False
                optimizer_lambda = Optimizer(filter(lambda p: p.requires_grad, safe_model.parameters()))
                train_safe_epoch(epoch+1, view, safe_model, data_loader, criterion_safe, optimizer_lambda, loss_record,
                                 device)

        res = valid(safe_model, device, dataset, view)
        for i in range(4):
            records["safe"][i].append(res[i])

        state = safe_model.state_dict()
        torch.save(state, './models/' + args.dataset + '/' + str(t) + '.pth')

    ind_ = np.argmin(np.array(loss_record["safe"]))
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f}'.format(records["safe"][0][ind_],
                                                        records["safe"][1][ind_],
                                                        records["safe"][3][ind_]))


if __name__ == '__main__':
    main()
