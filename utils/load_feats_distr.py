"""
To load and sample from the distribution of real features
"""
import numpy as np
import torch


class RealFeatureDistribution:
    """ A class to sample from the distribution of real features """
    def __init__(self, path):
        self.path = path
        self.probs, self.bin_edges = self.load_distribution(path)
        self.num_dims = len(self.probs)

    def load_distribution(self, path):
        """ Load the distribution of real features from a file """
        distr_dict = np.load(path, allow_pickle=True)
        probs = distr_dict.item().get('probs')
        bin_edges = distr_dict.item().get('bin_edges')
        return probs, bin_edges

    def sample_one_feat_dim(self, probs, bin_edges, batch_size=1):
        """ Sample one number from the distribution of a specific feature dimension
        :arg probs: the probabilities of the bins, for the feature dimension that you want to sample from
        :arg bin_edges: the bin edges of the bins, for the feature dimension that you want to sample from
        """
        bin_idx = np.random.choice(len(probs), size=batch_size, p=probs)
        edge_left = bin_edges[bin_idx]
        edge_right = bin_edges[bin_idx+1]
        return np.random.uniform(edge_left, edge_right)

    def sample(self, num_feats, batch_size=1):
        """ Sample n features from the distribution
        start from feat_dim 0, go towards the last real feat_dim |state_space|-1,
        then keep cycling through the real feat_dims until you have n samples
        :return an 2D array (torch tensor) of sampled features, size (batch_size, num_feats)
        """
        generated_feats = np.empty((batch_size, num_feats))
        for i in range(num_feats):
            feat_dim = i % self.num_dims
            feat = self.sample_one_feat_dim(self.probs[feat_dim], self.bin_edges[feat_dim], batch_size)
            generated_feats[:, i] = feat.transpose()
        return torch.from_numpy(generated_feats).to(torch.float32)


if __name__ == '__main__':
    feats_distr_folder = "../experiments/plots_fake_feats/real_feats_distributions/"
    feats_distr_file = "real_feats_distr_HalfCheetah.npy"
    feats_sampler = RealFeatureDistribution(path=f'{feats_distr_folder}{feats_distr_file}')
    sampled_feats = feats_sampler.sample(5, 3)
    print(sampled_feats)















