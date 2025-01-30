from kde import KNNKDE
import torch
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import time


def main():
    data = torch.randn(500000, 10, device="cuda").to(torch.float32)
    kde = KNNKDE(k=5, bandwidth=0.2, kernel="gaussian")
    kde_sklearn = KernelDensity(bandwidth=0.2, kernel="gaussian", rtol=0.1)
    data_cpu = data.cpu().numpy()
    kde_sklearn.fit(data_cpu)
    kde.fit(data)
    for _ in range(10):
        kde_densities = kde.kernel_density(data[0:20])

    start = time.time()
    knn_densities = kde.kernel_density(data)
    end = time.time()

    print(f"KDE time: {end - start}")
    start = time.time()
    scipy_densities = kde_sklearn.score_samples(data_cpu)
    end = time.time()
    print(f"Scipy time: {end - start}")

    data = data.cpu().numpy()
    knn_densities = knn_densities.cpu().numpy()

    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig("data.png")
    plt.close()

    plt.scatter(data[:, 0], data[:, 1], c=knn_densities)
    plt.savefig("knn_densities.png")
    plt.close()

    plt.scatter(data[:, 0], data[:, 1], c=scipy_densities)
    plt.savefig("scipy_densities.png")
    plt.close()



if __name__ == "__main__":
    main()