import torch
from sklearn.cluster import MiniBatchKMeans
from lib.config import cfg

class OnlineGaussianKMeans:
    def __init__(self, n_clusters, feature_dim, device='cpu'):
        self.n_clusters = n_clusters
        self.device = device
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=cfg.dtask.batch_size)
        # store counts, means, covariances
        self.counts = torch.zeros(n_clusters, device=device)
        self.means = torch.zeros(n_clusters, feature_dim, device=device)
        self.covs = torch.zeros(n_clusters, feature_dim, feature_dim, device=device)

    @torch.no_grad()
    def update(self, features):
        """
        features: torch.Tensor, shape [B, D]
        """
        feats_np = features.detach().cpu().numpy()
        self.kmeans.partial_fit(feats_np)
        labels = self.kmeans.predict(feats_np)
        
        for i in range(self.n_clusters):
            cluster_feats = features[labels == i]
            n_new = cluster_feats.shape[0]
            if n_new == 0:
                continue

            # old stats
            mu_old = self.means[i]
            sigma_old = self.covs[i]
            N_old = self.counts[i]

            # new batch stats
            mu_batch = cluster_feats.mean(dim=0)
            sigma_batch = ((cluster_feats - mu_batch).T @ (cluster_feats - mu_batch)) / n_new

            # update counts
            N_new = N_old + n_new

            # online mean update
            mu_new = (N_old * mu_old + n_new * mu_batch) / N_new

            # online covariance update
            sigma_new = (N_old * sigma_old + n_new * sigma_batch
                        + (N_old * n_new / N_new) * ((mu_old - mu_batch).unsqueeze(1) @ (mu_old - mu_batch).unsqueeze(0))
                        ) / N_new

            # store updates
            self.means[i] = mu_new
            self.covs[i] = sigma_new
            self.counts[i] = N_new
