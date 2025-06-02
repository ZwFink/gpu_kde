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

class IdentityScaler:
    def fit(self, X):
        pass

    def transform(self, X):
        return X
    


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
    Supports IndexFlatL2 and IndexIVFScalarQuantizer.

    Parameters:
        k: int, number of neighbors
        bandwidth: float, str, or torch.Tensor
            If float: uses the same bandwidth for all dimensions
            If 'scott': uses Scott's rule with dimension correction
            If torch.Tensor: uses different bandwidth for each dimension
        kernel: str, kernel type (only 'gaussian' supported)
        index_type: str, 'flat' or 'ivf_sq'. Default 'flat'.
        nlist: int, number of Voronoi cells for IVF index. Default 100.
        nprobe: int, number of cells to search for IVF index. Default 1.
    """
    def __init__(self, k=5, bandwidth=1.0, kernel="gaussian", index_type="ivf_sq", nlist=1000, nprobe=1):
        self.k = k
        self.bandwidth_spec = bandwidth
        self.bandwidth = None
        assert kernel.lower() == "gaussian", "Only Gaussian kernel is supported"
        self.kernel = kernel
        self.index_type = index_type.lower()
        assert self.index_type in ['flat', 'ivf_sq'], "index_type must be 'flat' or 'ivf_sq'"
        self.nlist = nlist
        self.nprobe = nprobe
        self.index = None
        self.gpu_available = faiss.get_num_gpus() > 0
        self.scaler = StandardScaler()
        self._X_scaled_stored = None # To store scaled data for reconstruction workaround
        self._X_device = None # Store device of original data

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
        3. Builds and trains (if necessary) the FAISS index
        4. Adds the data to the index
        5. Stores the scaled data for manual reconstruction workaround.
        """
        X = X.to(torch.float32)
        self._X_device = X.device # Store the original device
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        n_samples, n_features = X_scaled.shape

        # Set bandwidth based on specification
        self._set_bandwidth(X)

        # Create CPU index first
        if self.index_type == 'flat':
            cpu_index = faiss.IndexFlatL2(n_features)
        elif self.index_type == 'ivf_sq':
            quantizer = faiss.IndexFlatL2(n_features)
            # Using QT_8bit for potentially better stability with useFloat16=False on GPU
            cpu_index = faiss.IndexIVFScalarQuantizer(quantizer, n_features, self.nlist, faiss.ScalarQuantizer.QT_8bit)
            X_scaled_cpu = X_scaled.cpu()
            cpu_index.train(X_scaled_cpu)
            del X_scaled_cpu
            assert cpu_index.is_trained

        # Handle GPU transfer and add data
        if self.gpu_available:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False # Matches QT_8bit usage well
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            self.index.add(X_scaled)
            self._X_scaled_stored = X_scaled.clone() # Store on GPU
            self._gpu_mem_use = res.getMemoryInfo()
        else:
            self.index = cpu_index
            self.index.add(X_scaled)
            self._X_scaled_stored = X_scaled.cpu().clone() # Store on CPU

    def get_gpu_memory_use_mb(self):
        if self.gpu_available:
            mem_use_copy = self._gpu_mem_use.copy()[0]
            try:
                del mem_use_copy['TemporaryMemoryBuffer']
            except KeyError:
                pass
            finally:
                return (sum(map(lambda x: x[1], mem_use_copy.values()))) / (1024 * 1024)
        else:
            return 0

    def kernel_density(self, x, batch_size=32768):
        """Estimate the probability density at the given points, processing in batches.
        Uses manual reconstruction workaround if index is IVF_SQ.
        """
        n_samples = x.shape[0]
        if n_samples == 0:
            return torch.empty(0, device=x.device)

        assert self._X_scaled_stored is not None, "Model must be fit before calling kernel_density"
        # Ensure input x is on the same device as the original training data was
        x = x.to(self._X_device)

        assert not torch.isnan(x).any(), "Input 'x' contains NaNs"
        assert not torch.isinf(x).any(), "Input 'x' contains Infs"

        results = []

        if self.index_type == 'ivf_sq':
            if hasattr(self.index, 'nprobe'):
                 self.index.nprobe = self.nprobe
            elif hasattr(faiss, 'GpuParameterSpace'):
                 params = faiss.GpuParameterSpace()
                 params.set_index_parameter(self.index, 'nprobe', self.nprobe)

        for i in range(0, n_samples, batch_size):
            batch_start_index = i
            batch_end_index = min(i + batch_size, n_samples)

            x_batch = x[batch_start_index : batch_end_index].to(torch.float32)

            x_batch_scaled = self.scaler.transform(x_batch)

            D_batch, I_batch = self.index.search(x_batch_scaled, k=self.k)

            I_clamped = I_batch.clone().clamp(min=0)
            batch_size_cur = x_batch_scaled.shape[0]
            I_flat = I_clamped.view(-1)
            R_flat = self._X_scaled_stored[I_flat]
            R_batch = R_flat.view(batch_size_cur, self.k, -1)

            kernel_values_batch = _gaussian_kernel(x_batch_scaled, R_batch, self.bandwidth)

            # Replace mean density with sum over valid neighbors divided by count
            mask = (I_batch != -1).to(kernel_values_batch.dtype)
            sum_kernel = (kernel_values_batch * mask).sum(dim=1)
            valid_counts = mask.sum(dim=1).clamp(min=1)
            batch_density = sum_kernel / valid_counts
            results.append(batch_density)

        final_density = torch.cat(results, dim=0)

        return final_density

    def __getstate__(self):
        """Prepare the object state for serialization. Moves index to CPU if needed."""
        state = self.__dict__.copy()
        # Serialize index
        index_to_serialize = self.index
        if hasattr(faiss, 'GpuIndex') and isinstance(self.index, faiss.GpuIndex):
             index_to_serialize = faiss.index_gpu_to_cpu(self.index)
        state['index'] = faiss.serialize_index(index_to_serialize)

        # Handle _X_scaled_stored (move to CPU before saving)
        if state['_X_scaled_stored'] is not None:
            state['_X_scaled_stored'] = state['_X_scaled_stored'].cpu()

        return state

    def __setstate__(self, state):
        """Restore the object state from serialized data. Moves index and stored data to GPU if available."""
        index_bytes = state.pop('index')
        X_scaled_stored_cpu = state.pop('_X_scaled_stored', None) # Pop stored data

        # Restore the rest of the attributes first
        self.__dict__.update(state)

        # Deserialize index to CPU first
        cpu_index = faiss.deserialize_index(index_bytes)

        # Restore stored data and move to appropriate device
        if X_scaled_stored_cpu is not None:
            if self.gpu_available:
                self._X_scaled_stored = X_scaled_stored_cpu.to(self._X_device) # Use stored device info
            else:
                self._X_scaled_stored = X_scaled_stored_cpu
        else:
            self._X_scaled_stored = None

        # Move index to GPU if applicable
        if self.gpu_available:
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = False # Consistent with fit
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
        else:
            self.index = cpu_index

        # Set nprobe for IVF index after loading
        if self.index_type == 'ivf_sq':
             if hasattr(self.index, 'nprobe'):
                 self.index.nprobe = self.nprobe
             elif hasattr(faiss, 'GpuParameterSpace'):
                 params = faiss.GpuParameterSpace()
                 params.set_index_parameter(self.index, 'nprobe', self.nprobe)
