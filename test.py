import torch
import argparse
from dataloader import load_data
from torch.utils.data import DataLoader
import numpy as np
from evaluation import evaluate
from model import DSMVC


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--dataset', default='caltech_5m')
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=240)
parser.add_argument("--tune_epochs", default=30)
parser.add_argument("--feature_dim", default=256)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--epochs", default=120)
parser.add_argument("--view", type=int, default=2)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(acc, nmi, pur))


dataset, dims, view, data_size, class_num = load_data(args.dataset)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
safe_train_epoch = args.epochs
view_num = args.view

safe_model = DSMVC(view_num-1, view_num, dims, args.feature_dim, class_num)
safe_model = safe_model.to(device)
if args.dataset in ['uci', 'caltech_5m']:
    checkpoint = torch.load('./models/' + args.dataset + '/' + str(args.view) + '/model.pth')
else:
    checkpoint = torch.load('./models/' + args.dataset + '/model.pth')
safe_model.load_state_dict(checkpoint)
valid(safe_model, device, dataset, view)
