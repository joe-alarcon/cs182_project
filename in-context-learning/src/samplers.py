import math

import torch


class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    names_to_classes = {
        "gaussian": GaussianSampler,
        "nums": Base2BaseSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b

class Base2BaseSampler(DataSampler):
    def __init__(self, n_dims):
        super().__init__(n_dims)

    def generating_sequence(self, base_in, base_out, n, lower, upper):
        def convert_base(n, base):
            if not (2 <= base <= 36):
                raise ValueError("Base must be between 2 and 36")
            digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if n == 0:
                return "0"
            sign = "-" if n < 0 else ""
            n = abs(n)
            result = []
            while n > 0:
                result.append(digits[n % base])
                n //= base
            return int(sign + ''.join(reversed(result)))
    
        rdm_nums = torch.randint(lower, high=upper, size=(n,1))
        ins = [convert_base(rdm_num, base_in) for rdm_num in rdm_nums]
        outs = [convert_base(rdm_num, base_out) for rdm_num in rdm_nums]
        return torch.tensor([ins]).t(), torch.tensor([outs]).t().view((n,))
    
    def sample_xs(self, n_points, b_size, n_dims_truncated, lower=1, upper=1000, seeds=None):
        if seeds is not None:
            generator = torch.Generator()
            generator.manual_seed(seeds)
        xs = torch.zeros(b_size, n_points, self.n_dims)
        ys = torch.zeros(b_size, n_points)
        # base_in = 2
        for i in range(b_size):
            # TODO: change base_in for sampling within a batch, base_out = 10
            base_in = torch.randint(2,9)
            xs[i], ys[i] = self.generating_sequence(base_in, 10, n_points, lower, upper)
        return xs, ys