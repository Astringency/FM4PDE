import h5py
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from pdebench.data_gen.src.sim_radial_dam_break import RadialDamBreak2D

def process_batch(batch_start, batch_size):
    batch_id = batch_start // batch_size
    save_path = "/large_storage/zhangxf/PDEdata/test1125/"
    for i in tqdm(range(batch_size), desc=f"Batch {batch_id+1}"):
        seed = batch_start + i
        rng = np.random.default_rng(seed)
        # dam_radius = rng.uniform(0.3, 0.7)
        # inner_height = rng.uniform(1.5, 2.5)
        dam_radius = rng.uniform(0.4, 0.8)
        inner_height = rng.uniform(2, 3)
        
        swe = RadialDamBreak2D(
            xdim=128,
            ydim=128,
            grav=1.0,
            dam_radius=dam_radius,
            inner_height=inner_height
        )
        
        swe.run(T=1.0, tsteps=10)
        
        # file_name = f"{save_path}2d_swe_128_128_10_{batch_id}.h5"
        file_name = f"{save_path}swe_test_1000-128-128-10.h5"
        seed_str = str(seed).zfill(5)
        
        with h5py.File(file_name, "a") as f:
            swe.save_state_to_disk(f, seed_str)
            seed_group = f[seed_str]
            seed_group.attrs["xdim"] = swe.xdim
            seed_group.attrs["ydim"] = swe.ydim
            seed_group.attrs["grav"] = swe.grav
            seed_group.attrs["dam_radius"] = dam_radius
            seed_group.attrs["inner_height"] = inner_height
            seed_group.attrs["x_range"] = (swe.xlower, swe.xupper)
            seed_group.attrs["y_range"] = (swe.ylower, swe.yupper)
        
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
    # print(f"All {total_samples} samples done.")

    # === #
    # generate test data
    total_samples = 1000
    process_batch(0, total_samples)

    print(f"All {total_samples} samples done.")

