import torch
import torch.nn as nn
from kernel import *
EPSILON = 1E-9


class Loss(nn.Module):
    def __init__(self, batch_size, class_num, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.device = device

    def forward_cluster(self, hidden, output, print_sign=False):
        hidden_kernel = vector_kernel(hidden, rel_sigma=0.15)
        l1 = self.DDC1(output, hidden_kernel, self.class_num)
        l2 = self.DDC2(output)
        l3 = self.DDC3(self.class_num, output, hidden_kernel)
        if print_sign:
            print(l1.item())
            print(l2.item())
            print(l3.item())
        return l1+l2+l3, l1.item() + l2.item() + l3.item()

    "Adopted from https://github.com/DanielTrosten/mvc"

    def triu(self, X):
        # Sum of strictly upper triangular part
        return torch.sum(torch.triu(X, diagonal=1))

    def _atleast_epsilon(self, X, eps=EPSILON):
        """
        Ensure that all elements are >= `eps`.

        :param X: Input elements
        :type X: th.Tensor
        :param eps: epsilon
        :type eps: float
        :return: New version of X where elements smaller than `eps` have been replaced with `eps`.
        :rtype: th.Tensor
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    def d_cs(self, A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.

        :param A: Cluster assignment matrix
        :type A:  th.Tensor
        :param K: Kernel matrix
        :type K: th.Tensor
        :param n_clusters: Number of clusters
        :type n_clusters: int
        :return: CS-divergence
        :rtype: th.Tensor
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(torch.diagonal(nom), 0)

        nom = self._atleast_epsilon(nom)
        dnom_squared = self._atleast_epsilon(dnom_squared, eps=EPSILON ** 2)

        d = 2 / (n_clusters * (n_clusters - 1)) * self.triu(nom / torch.sqrt(dnom_squared))
        return d

    # ======================================================================================================================
    # Loss terms
    # ======================================================================================================================

    def DDC1(self, output, hidden_kernel, n_clusters):
        """
        L_1 loss from DDC
        """
        # required_tensors = ["hidden_kernel"]
        return self.d_cs(output, hidden_kernel, n_clusters)

    def DDC2(self, output):
        """
        L_2 loss from DDC
        """
        n = output.size(0)
        return 2 / (n * (n - 1)) * self.triu(output @ torch.t(output))

    def DDC2Flipped(self, output, n_clusters):
        """
        Flipped version of the L_2 loss from DDC. Used by EAMC
        """

        return 2 / (n_clusters * (n_clusters - 1)) * self.triu(torch.t(output) @ output)

    def DDC3(self, n_clusters, output, hidden_kernel):
        """
        L_3 loss from DDC
        """

        eye = torch.eye(n_clusters, device=self.device)

        m = torch.exp(-cdist(output, eye))
        return self.d_cs(m, hidden_kernel, n_clusters)
