import faiss
import numpy as np
import torch
from typing import Union
from faiss.contrib import torch_utils
import time

TensorOrArray = Union[torch.Tensor, np.ndarray]


@torch.jit.script
def _gaussian_kernel(x: torch.Tensor, y: torch.Tensor, bandwidth: torch.Tensor) -> torch.Tensor:
    """Compute the Gaussian kernel between points x and their k-nearest neighbors y.

    Args:
        x (torch.Tensor): Input points of shape [batch_size, features]
        y (torch.Tensor): K-nearest neighbors of shape [batch_size, k, features]
        bandwidth (torch.Tensor): Kernel bandwidth for each dimension of shape [features]

    Returns:
        torch.Tensor: Kernel values of shape [batch_size, k]
    """
    batch_size, k, features = y.shape
    
    # Reshape x to [batch_size, 1, features] to broadcast with y
    x_expanded = x.unsqueeze(1)
    
    diff = x_expanded - y
    # Reshape bandwidth to [1, 1, features] for broadcasting
    bandwidth_expanded = bandwidth.view(1, 1, -1)
    squared_distances = torch.sum((diff / bandwidth_expanded) * (diff / bandwidth_expanded), dim=2)
    
    # Compute kernel values (no need to square bandwidth since we already divided by it)
    exponent = -0.5 * squared_distances
    
    return torch.exp(exponent)

@torch.jit.script
def _scaler_transform(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Transform features by standardizing to zero mean and unit variance.

    Args:
        x (torch.Tensor): Input tensor to transform
        mean (torch.Tensor): Mean of each feature
        std (torch.Tensor): Standard deviation of each feature

    Returns:
        torch.Tensor: Transformed tensor with zero mean and unit variance
    """
    return (x - mean) / std

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.
    
    Similar to sklearn's StandardScaler, but implemented for PyTorch tensors.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Compute the mean and std to be used for later scaling.

        Args:
            X (torch.Tensor): Training data to compute mean and std from
        """
        self.mean = X.mean(dim=0)
        self.std = X.std(dim=0)
        self.std = torch.clamp(self.std, min=1e-6)

    def transform(self, X):
        """Perform standardization by centering and scaling.

        Args:
            X (torch.Tensor): Data to transform

        Returns:
            torch.Tensor: Standardized data
        """
        return _scaler_transform(X, self.mean, self.std)

class KNNKDE:
    """
    K-Nearest Neighbors Kernel Density Estimation.
    Uses FAISS for efficient nearest neighbor search.
    L2 distance is used for the distance metric.

    Parameters:
        k: int, number of neighbors
        bandwidth: float, str, or torch.Tensor
            If float: uses the same bandwidth for all dimensions
            If 'scott': uses Scott's rule with dimension correction
            If torch.Tensor: uses different bandwidth for each dimension
        kernel: str, kernel type (only 'gaussian' supported)
    """
    def __init__(self, k=5, bandwidth=1.0, kernel="gaussian"):
        self.k = k
        self.bandwidth_spec = bandwidth  # Store the specification
        self.bandwidth = None  # Will be set during fit
        assert kernel.lower() == "gaussian", "Only Gaussian kernel is supported"
        self.kernel = kernel
        self.index = None
        self.gpu_available = faiss.get_num_gpus() > 0
        self.scaler = StandardScaler()

    def _compute_scott_bandwidth(self, n_samples: int, n_features: int, std: torch.Tensor) -> torch.Tensor:
        """Compute Scott's rule bandwidth with high-dimensional correction.

        Args:
            n_samples (int): Number of samples in the dataset
            n_features (int): Number of features/dimensions
            std (torch.Tensor): Standard deviation of each feature

        Returns:
            torch.Tensor: Computed bandwidth for each dimension
        """
        # Basic Scott's rule
        scott_factor = n_samples ** (-1.0 / (n_features + 4))
        
        # Apply dimension correction: as dimensions increase, we slow down the 
        # bandwidth decrease to prevent undersmoothing
        dim_correction = max(1.0, np.log10(n_features))
        corrected_bandwidth = std * scott_factor * dim_correction
        
        # Add safety clipping to prevent extreme values
        return torch.clamp(corrected_bandwidth, min=1e-3, max=100.0)

    def _set_bandwidth(self, X: torch.Tensor) -> None:
        """Set the kernel bandwidth based on the specification and input data.

        Args:
            X (torch.Tensor): Input data tensor of shape [n_samples, n_features]
        
        Raises:
            ValueError: If bandwidth specification is invalid
        """
        n_samples, n_features = X.shape
        device = X.device

        if isinstance(self.bandwidth_spec, str) and self.bandwidth_spec.lower() == 'scott':
            self.bandwidth = self._compute_scott_bandwidth(
                n_samples, 
                n_features, 
                self.scaler.std
            ).to(device)
        elif isinstance(self.bandwidth_spec, (int, float)):
            # Use same bandwidth for all dimensions
            self.bandwidth = torch.full(
                (n_features,), 
                float(self.bandwidth_spec),
                device=device
            )
        elif isinstance(self.bandwidth_spec, torch.Tensor):
            assert len(self.bandwidth_spec) == n_features, \
                f"Bandwidth tensor must have {n_features} elements"
            self.bandwidth = self.bandwidth_spec.to(device)
        else:
            raise ValueError(
                "bandwidth must be 'scott', a number, or a tensor"
            )

    def fit(self, X):
        """Fit the KNN-KDE model to the input data.

        This method:
        1. Fits the standard scaler
        2. Sets the bandwidth
        3. Builds the FAISS index for efficient nearest neighbor search

        Args:
            X (torch.Tensor): Training data of shape [n_samples, n_features]
        """
        X = X.to(torch.float32)
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        # Set bandwidth based on specification
        self._set_bandwidth(X)

        if self.gpu_available:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False   
            self.index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatL2(X_scaled.shape[1]), co)
        else:
            self.index = faiss.IndexFlatL2(X_scaled.shape[1])
        self.index.add(X_scaled)

    
    def kernel_density(self, x):
        """Estimate the probability density at the given points.

        Args:
            x (torch.Tensor): Points to estimate density at, shape [batch_size, features]

        Returns:
            torch.Tensor: Estimated density values of shape [batch_size]
        """
        # x can now be [batch_size, features]
        x = x.to(torch.float32)
        x = self.scaler.transform(x)
        D, I, R = self.index.search_and_reconstruct(x, k=self.k)
        kernel_values = _gaussian_kernel(x, R, self.bandwidth)
        return torch.mean(kernel_values, dim=1)  # Average over k neighbors for each sample

    # we need to save the index to the CPU
    # so we can be serialized
    def __getstate__(self):
        state = self.__dict__.copy()
        # check if the index is on the gpu
        state['index'] = faiss.serialize_index(faiss.index_gpu_to_cpu(self.index))
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = False
        self.index = faiss.index_cpu_to_gpu(res, 0, faiss.deserialize_index(self.index), co)
