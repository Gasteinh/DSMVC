from torch.utils.data import Dataset
import numpy as np
import torch


class MultiviewDataset(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            data.append(torch.tensor(self.data_list[i][idx]))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


class MultiviewDataset2(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            x = torch.tensor(self.data_list[i][idx])
            data.append(x.view(x.size()[0], 28, 28))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


def load_data(name):
    data_path = './data/'
    dataset_names = ['caltech_5m', 'uci', 'rgbd', 'voc', 'mnist_mv']
    if name in dataset_names:
        path = data_path + name + '.npz'
        data = np.load(path)
        num_views = int(data['n_views'])
        data_list = []
        for i in range(num_views):
            x = data[f"view_{i}"]
            if len(x.shape) > 2:
                x = x.reshape([x.shape[0], -1])
            data_list.append(x.astype(np.float32))
        labels = data['labels']
        dims = []
        for i in range(num_views):
            dims.append(data_list[i].shape[1])
        class_num = labels.max() + 1
        data_size = data_list[0].shape[0]
        dataset = MultiviewDataset(num_views, data_list, labels)
        return dataset, dims, num_views, data_size, class_num
    elif name == 'mnist_mv' or 'fmnist':
        path = data_path + name + '.npz'
        data = np.load(path)
        num_views = int(data['n_views'])
        data_list = []
        for i in range(num_views):
            x = data[f"view_{i}"]
            data_list.append(x)
        labels = data['labels']
        dims = []
        for i in range(num_views):
            dims.append(data_list[i].shape[1])
        class_num = labels.max() + 1
        data_size = data_list[0].shape[0]
        dataset = MultiviewDataset2(num_views, data_list, labels)
        return dataset, dims, num_views, data_size, class_num
    else:
        raise NotImplementedError
