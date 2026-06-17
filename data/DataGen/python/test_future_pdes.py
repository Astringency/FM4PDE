from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "data" / "DataGen" / "python" / "generate_future_pdes.py"


class FuturePDEGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="fm4pde_future_pdes_")
        cls.out_root = Path(cls.tmp.name)
        cmd = [
            sys.executable,
            str(GEN),
            "--pde",
            "all",
            "--out-root",
            str(cls.out_root),
            "--resolution",
            "16",
            "--n-train",
            "8",
            "--n-test",
            "4",
            "--train-shards",
            "2",
            "--n-time",
            "5",
            "--quick-test",
            "--overwrite",
        ]
        subprocess.run(cmd, cwd=ROOT, check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def test_hdf5_keys_shapes_and_dtype(self) -> None:
        expected_channels = {
            "heat": (1, 1, 2),
            "wave": (2, 2, 4),
            "advection_diffusion": (4, 4, 8),
        }
        for pde, (cin, cout, ctot) in expected_channels.items():
            path = self.out_root / pde / f"{pde}_4-16-16_1.h5"
            self.assertTrue(path.exists(), path)
            with h5py.File(path, "r") as h5:
                for key in ["input_data", "output_data", "data", "full_trajectory", "x", "y", "t", "sample_seed"]:
                    self.assertIn(key, h5)
                self.assertEqual(h5["input_data"].shape, (4, cin, 16, 16))
                self.assertEqual(h5["output_data"].shape, (4, cout, 16, 16))
                self.assertEqual(h5["data"].shape, (4, ctot, 16, 16))
                self.assertEqual(h5["full_trajectory"].shape, (4, 1, 5, 16, 16))
                self.assertEqual(h5["data"].dtype, np.dtype("float32"))

    def test_no_leakage_json_and_hashes(self) -> None:
        for pde in ["heat", "wave", "advection_diffusion"]:
            check_path = self.out_root / pde / "no_leakage_check.json"
            self.assertTrue(check_path.exists())
            check = json.loads(check_path.read_text(encoding="utf-8"))
            self.assertTrue(check["ok"], check)
            self.assertEqual(check["seed_overlap_count"], 0)
            self.assertEqual(check["input_hash_overlap_count"], 0)

    def test_heat_variance_decays(self) -> None:
        path = self.out_root / "heat" / "heat_4-16-16_1.h5"
        with h5py.File(path, "r") as h5:
            traj = h5["full_trajectory"][:, 0]
            v0 = np.var(traj[:, 0], axis=(1, 2))
            vT = np.var(traj[:, -1], axis=(1, 2))
            self.assertTrue(np.all(vT <= v0 + 1e-5))

    def test_wave_is_finite_and_reasonable(self) -> None:
        path = self.out_root / "wave" / "wave_4-16-16_1.h5"
        with h5py.File(path, "r") as h5:
            data = h5["data"][:]
            self.assertTrue(np.isfinite(data).all())
            self.assertLess(float(np.max(np.abs(data))), 20.0)

    def test_advection_diffusion_is_finite_and_smooths(self) -> None:
        path = self.out_root / "advection_diffusion" / "advection_diffusion_4-16-16_1.h5"
        with h5py.File(path, "r") as h5:
            traj = h5["full_trajectory"][:, 0]
            self.assertTrue(np.isfinite(traj).all())
            fft0 = np.fft.fft2(traj[:, 0], axes=(-2, -1))
            fftT = np.fft.fft2(traj[:, -1], axes=(-2, -1))
            freq = np.fft.fftfreq(16)
            kx, ky = np.meshgrid(freq, freq, indexing="ij")
            high = (kx**2 + ky**2) > 0.10
            e0 = np.mean(np.abs(fft0[:, high]) ** 2, axis=1)
            eT = np.mean(np.abs(fftT[:, high]) ** 2, axis=1)
            self.assertTrue(np.all(eT <= e0 + 1e-5))

    def test_loader_reads_quick_train_shards(self) -> None:
        sys.path.insert(0, str(ROOT))
        from data.load import PDEloader

        expected = {"heat": (8, 2, 7.0), "wave": (8, 4, 8.0), "advection_diffusion": (8, 8, 9.0)}
        for pde, (n, channels, label_value) in expected.items():
            data, label = PDEloader(pde).load_data(str(self.out_root) + "/", size=2)
            self.assertEqual(tuple(data.shape), (n, channels, 16, 16))
            self.assertEqual(tuple(label.shape), (n,))
            self.assertEqual(float(label[0]), label_value)

    def test_compileall_data(self) -> None:
        subprocess.run([sys.executable, "-m", "compileall", "-q", "data"], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()

