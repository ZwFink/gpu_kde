from torch.utils.data import Dataset
import torch
import kde
import re
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import time

percentile_re = re.compile(r'(?:\[(\d+),\s{0,1}(\d+)\],{0,1})')

class DatasetCommon():
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__
        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self._apply_slice()
            self._percentile_partition()
            self._dtype_conversion()
        cls.__init__ = new_init


    def __len__(self):
        return self.len

    def to(self, device):
        self.input = self.input.to(device)
        self.output = self.output.to(device)
        return self

    @property
    def len(self):
        return len(self.input)

    def __getitem__(self, idx):
        return (self.input[idx],
                self.output[idx])

    def input_as_torch_tensor(self):
        return self.input

    def output_as_torch_tensor(self):
        return self.output

    def get_percentiles(self):
        try:
            percs = self.kwargs['percentiles']
            parsed = percentile_re.findall(percs)
            opt_percs = list()
            for p in parsed:
                lower, uper = int(p[0]), int(p[1])
                opt_percs.append((lower, uper))
            return opt_percs
        except KeyError:
            return [(0, 100)]


    def percentile_partition(self, percentiles):
        input_tensor = self.input_as_torch_tensor()
        output_tensor = self.output_as_torch_tensor()

        if len(output_tensor.shape) > 2:
            return input_tensor, output_tensor
        
        unique_percentiles = sorted(set(p for range_pair in percentiles for p in range_pair))
        percentile_values = torch.tensor([
            torch.quantile(output_tensor, q/100) for q in unique_percentiles
        ])
        
        percentile_dict = dict(zip(unique_percentiles, percentile_values))
        
        mask = torch.zeros(len(output_tensor), dtype=torch.bool)

        for lower, upper in percentiles:
            lower_value = percentile_dict[lower]
            upper_value = percentile_dict[upper]
            if lower == 0:
                mask |= (output_tensor <= upper_value).view(len(output_tensor))
            else:
                mask |= ((output_tensor > lower_value) & (output_tensor <= upper_value)).view(len(output_tensor))
        
        partitioned_input = input_tensor[mask]
        partitioned_output = output_tensor[mask]
        
        return partitioned_input, partitioned_output


    def _percentile_partition(self):
        self.input, self.output = self.percentile_partition(self.get_percentiles())

    def _dtype_conversion(self):
        try:
            dt = self.kwargs['dtype']
            self.input = self.input.type(dtype=getattr(torch, dt))
            self.output = self.output.type(dtype=getattr(torch, dt))
        except KeyError:
            pass

    def _apply_slice(self):
        try:
            subset = self.kwargs['subset']
            if 'step' not in subset:
                subset['step'] = 1
            if 'start' not in subset:
                subset['start'] = 0

            start = subset['start']
            stop = subset['stop']
            step = subset['step']
            slc = slice(start, stop, step)
            self.input = self.input[slc]
            self.output = self.output[slc]
        except KeyError:
            pass

    @property
    def dtype(self):
        return self.input.dtype

    def train_test_split(self, test_proportion: float):
        test_size = int(len(self) * test_proportion)
        train_size = len(self) - test_size
        return torch.utils.data.random_split(self, [train_size, test_size])

class ARFFDataSet(DatasetCommon, Dataset):
    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)
        self.path = path
        self.input, self.output = self.read_arff_file(path)
        self.input, self.output = torch.tensor(self.input), torch.tensor(self.output)

    def read_arff_file(self, path):
        from scipy.io import arff
        import pandas as pd
        data, meta = arff.loadarff(path)
        df = pd.DataFrame(data)
        return df.iloc[:, :-1].values, np.expand_dims(df.iloc[:, -1].values, -1)

    @property
    def shape(self):
        return self.input.shape


def main():
    ailerons_path = '/scratch/mzu/zanef2/surrogates/SurrogateResults/2024-08-09_kde_new_datasets/ailerons.arff'
    data = ARFFDataSet(ailerons_path)
    print(data.input.shape)
    print(data.output.shape)

    id_tails_percentiles = [(0, 70)]
    ood_tails_percentiles = [(70, 100)]
    id_tails_data = data.percentile_partition(id_tails_percentiles)
    ood_tails_data = data.percentile_partition(ood_tails_percentiles)

    id_input, id_output = id_tails_data
    ood_input, ood_output = ood_tails_data
    id_input = id_input.to(torch.float32).to("cuda")
    ood_input = ood_input.to(torch.float32).to("cuda")
    id_output = id_output.to(torch.float32).to("cuda")
    ood_output = ood_output.to(torch.float32).to("cuda")

    gpu_kde = kde.KNNKDE(k=15, bandwidth='scott', kernel="gaussian")
    gpu_kde.fit(id_input)
    kde_sklearn = KernelDensity(bandwidth=0.2, kernel="gaussian", rtol=0.1)
    kde_sklearn.fit(id_input.cpu().numpy())

    # do some warmups for the jit
    for _ in range(10):
        kde_densities = gpu_kde.kernel_density(id_input)

    start_time = time.time()
    kde_densities = gpu_kde.kernel_density(id_input)
    end_time = time.time()
    print(f"KDE time: {end_time - start_time}")
    kde_densities_ood = gpu_kde.kernel_density(ood_input)

    id_input_cpu = id_input.cpu().numpy()
    start_time = time.time()
    kde_sklearn_densities = kde_sklearn.score_samples(id_input_cpu)
    end_time = time.time()
    print(f"Scipy time: {end_time - start_time}")
    ood_input_cpu = ood_input.cpu().numpy()
    kde_sklearn_densities_ood = kde_sklearn.score_samples(ood_input_cpu)

    id_input = id_input.cpu().numpy()
    ood_input = ood_input.cpu().numpy()
    kde_densities = kde_densities.cpu().numpy()
    kde_densities_ood = kde_densities_ood.cpu().numpy()
    kde_sklearn_densities = kde_sklearn_densities
    kde_sklearn_densities_ood = kde_sklearn_densities_ood
    #normalize our densities to [0, 1] so they are on the same scale
    # kde_densities and kde_densities_ood should be on the same scale
    # kde_sklearn_densities and kde_sklearn_densities_ood should be on the same scale
    # kde_densities = (kde_densities - np.min(kde_densities)) / (np.max(kde_densities) - np.min(kde_densities))
    # kde_densities_ood = (kde_densities_ood - np.min(kde_densities_ood)) / (np.max(kde_densities) - np.min(kde_densities))
    # kde_sklearn_densities = (kde_sklearn_densities - np.min(kde_sklearn_densities)) / (np.max(kde_sklearn_densities) - np.min(kde_sklearn_densities))
    # kde_sklearn_densities_ood = (kde_sklearn_densities_ood - np.min(kde_sklearn_densities_ood)) / (np.max(kde_sklearn_densities) - np.min(kde_sklearn_densities))

    # we need to do an eCDF of the kde_densities and the scipy_densities and plot them
    plt.ecdf(kde_densities, label="KDE")
    # plt.ecdf(kde_sklearn_densities, label="Scipy")
    plt.ecdf(kde_densities_ood, label="KDE OOD")
    # plt.ecdf(kde_sklearn_densities_ood, label="Scipy OOD")
    plt.legend()
    plt.savefig("densities.png")
    plt.close()

if __name__ == "__main__":
    main()