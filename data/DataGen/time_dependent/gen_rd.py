import h5py
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pdebench.data_gen.src.sim_diff_react import Simulator as DiffReactSimulator

def process_batch(batch_start, batch_size):
    batch_id = batch_start // batch_size
    # save_path = "/large_storage/zhangxf/PDEdata/reaction_diffusion/"
    save_path = "/large_storage/zhangxf/PDEdata/test1125/"
    os.makedirs(save_path, exist_ok=True)
    
    for i in tqdm(range(batch_size), desc=f"Batch {batch_id+1}"):
        seed = batch_start + i
        rng = np.random.default_rng(seed)
        
        # Du = rng.uniform(0.001, 0.01)
        # Dv = rng.uniform(0.005, 0.05)
        # k = rng.uniform(0.002, 0.02)
        # Du = 1e-3
        # Dv = 5e-3
        # k = 5e-3
        
        Du = 2e-3
        Dv = 4e-3
        k = 3e-3

        diff_react = DiffReactSimulator(
            xdim = 128,
            ydim = 128,
            Du = Du,
            Dv = Dv,
            k = k,
            t = 5,
            tdim = 100,
            x_left = -1.0,
            x_right = 1.0,
            y_bottom = -1.0,
            y_top = 1.0,
            seed = seed
        )
        
        data_sample = diff_react.generate_sample()
        
        # file_name = f"{save_path}reaction_diffusion-128-128-100_{batch_id}.h5"
        file_name = f"{save_path}reaction_diffusion_test_1000-128-128-10.h5"
        seed_str = str(seed).zfill(5)
        
        with h5py.File(file_name, "a") as f:
            f.create_dataset(
                f"{seed_str}/data",
                data=data_sample,
                dtype="float32",
                compression="lzf"
            )
            f.create_dataset(f"{seed_str}/grid/x", data=diff_react.x, dtype="float32")
            f.create_dataset(f"{seed_str}/grid/y", data=diff_react.y, dtype="float32")
            f.create_dataset(f"{seed_str}/grid/t", data=diff_react.t, dtype="float32")
            
            seed_group = f[seed_str]
            seed_group.attrs["xdim"] = 128
            seed_group.attrs["ydim"] = 128
            seed_group.attrs["tdim"] = 100
            seed_group.attrs["Du"] = Du
            seed_group.attrs["Dv"] = Dv
            seed_group.attrs["k"] = k
            seed_group.attrs["x_range"] = (-1.0, 1.0)
            seed_group.attrs["y_range"] = (-1.0, 1.0)
            seed_group.attrs["T"] = 5.0

if __name__ == "__main__":
    # generate training data
    # total_samples = 50000
    # processes = 5
    # samples_per_process = total_samples // processes
    #
    # batch_starts = [i * samples_per_process for i in range(processes)]
    #
    # with mp.Pool(processes=processes) as pool:
    #     pool.starmap(process_batch, [(start, samples_per_process) for start in batch_starts])
    #
    # print(f"All {total_samples} reaction diffusion samples generated.")

    # === #

    # generate test data
    total_samples = 1000
    process_batch(0, total_samples)
