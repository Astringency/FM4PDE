import numpy as np
import pickle


if __name__ == "__main__":

    sample_path = "/research_data/users/zhangxifeng/C01Python/Eval/DeepLearning4PDE/samples/FM4PDE/both/reaction_diffusion_sparse_0_results.pkl"

    with open(sample_path, 'rb') as f:
        data = pickle.load(f)

    print(data.keys())
