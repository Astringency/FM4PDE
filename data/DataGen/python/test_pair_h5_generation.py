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
GEN = ROOT / "data" / "DataGen" / "python" / "generate_pair_h5s.py"


class PairH5GenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="fm4pde_pair_h5_")
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

    def test_hdf5_on_disk_schema(self) -> None:
        expected = {
            "heat": {"cin": 1, "cout": 1, "scalars": ["alpha"], "trajectory_channels": 1},
            "wave": {"cin": 2, "cout": 2, "scalars": [], "trajectory_channels": 2},
            "advection_diffusion": {"cin": 1, "cout": 1, "scalars": ["b_x", "b_y", "kappa"], "trajectory_channels": 1},
            "steady_heat_conduction": {
                "cin": 1,
                "cout": 1,
                "scalars": ["u_D", "residual_norm", "picard_iters", "converged"],
                "trajectory_channels": 0,
            },
        }
        for pde, spec in expected.items():
            path = self.out_root / pde / f"{pde}_4-16-16_1.h5"
            self.assertTrue(path.exists(), path)
            with h5py.File(path, "r") as h5:
                self.assertIn("input_data", h5)
                self.assertIn("output_data", h5)
                self.assertNotIn("data", h5)
                self.assertNotIn("materialized_data", h5)
                self.assertEqual(h5["input_data"].shape, (4, spec["cin"], 16, 16))
                self.assertEqual(h5["output_data"].shape, (4, spec["cout"], 16, 16))
                self.assertEqual(h5["input_data"].dtype, np.dtype("float32"))
                for key in spec["scalars"]:
                    self.assertIn(key, h5)
                    self.assertEqual(h5[key].shape[0], 4)
                if spec["trajectory_channels"]:
                    self.assertEqual(
                        h5["full_trajectory"].shape,
                        (4, spec["trajectory_channels"], 5, 16, 16),
                    )
                else:
                    self.assertNotIn("full_trajectory", h5)

    def test_heat_alpha_metadata_and_decay(self) -> None:
        sys.path.insert(0, str(ROOT))
        from data.load import PDEloader

        path = self.out_root / "heat" / "heat_4-16-16_1.h5"
        with h5py.File(path, "r") as h5:
            self.assertEqual(h5["alpha"].shape, (4,))
            traj = h5["full_trajectory"][:, 0]
            v0 = np.var(traj[:, 0], axis=(1, 2))
            vT = np.var(traj[:, -1], axis=(1, 2))
            self.assertTrue(np.all(vT <= v0 + 1e-5))
            self.assertTrue(np.isfinite(h5["output_data"][:]).all())
        data, label, metadata = PDEloader("heat").load_data(str(self.out_root) + "/", size=2, return_metadata=True)
        self.assertEqual(tuple(data.shape), (8, 2, 16, 16))
        self.assertEqual(float(label[0]), 7.0)
        self.assertEqual(metadata["channel_names"], ["u0", "uT"])
        self.assertIn("alpha", metadata["pde_params"])
        self.assertEqual(tuple(metadata["pde_params"]["alpha"].shape), (8,))

    def test_wave_vt_and_fixed_c_schema(self) -> None:
        path = self.out_root / "wave" / "wave_4-16-16_1.h5"
        with h5py.File(path, "r") as h5:
            self.assertNotIn("c", h5)
            self.assertIn("fixed_c", h5.attrs)
            self.assertNotIn("c_field", h5)
            v0 = h5["input_data"][:, 1]
            vT = h5["output_data"][:, 1]
            self.assertTrue(np.isfinite(vT).all())
            self.assertTrue(np.allclose(v0, 0.0))
            self.assertGreater(float(np.max(np.abs(vT))), 1e-4)

    def test_advection_diffusion_scalar_metadata_and_sign(self) -> None:
        sys.path.insert(0, str(ROOT))
        from data.load import PDEloader
        from data.DataGen.python.common import periodic_wavenumbers

        path = self.out_root / "advection_diffusion" / "advection_diffusion_4-16-16_1.h5"
        with h5py.File(path, "r") as h5:
            self.assertEqual(h5["input_data"].shape, (4, 1, 16, 16))
            self.assertEqual(h5["output_data"].shape, (4, 1, 16, 16))
            self.assertEqual(h5["b_x"].shape, (4,))
            self.assertEqual(h5["b_y"].shape, (4,))
            self.assertEqual(h5["kappa"].shape, (4,))
            self.assertTrue(np.all(h5["kappa"][:] > 0.0))
        data, _, metadata = PDEloader("advection_diffusion").load_data(
            str(self.out_root) + "/",
            size=2,
            return_metadata=True,
        )
        self.assertEqual(tuple(data.shape), (8, 2, 16, 16))
        self.assertEqual(metadata["channel_names"], ["u0", "uT"])
        self.assertEqual(set(metadata["pde_params"]), {"b_x", "b_y", "kappa", "T"})
        self.assertEqual(tuple(metadata["pde_params"]["kappa"].shape), (8,))

        s = 32
        x = np.arange(s) / s
        xx, _ = np.meshgrid(x, x, indexing="ij")
        u0 = np.cos(2.0 * np.pi * xx)
        bx, by, kappa, t = 0.25, 0.0, 0.0, 0.5
        kx, ky, ksq = periodic_wavenumbers(s)
        evolved = np.fft.ifft2(np.fft.fft2(u0) * np.exp(-(kappa * ksq + 1j * (bx * kx + by * ky)) * t)).real
        expected = np.cos(2.0 * np.pi * (xx - bx * t))
        self.assertLess(float(np.max(np.abs(evolved - expected))), 1e-10)

    def test_physical_wavenumber_scale(self) -> None:
        sys.path.insert(0, str(ROOT))
        from data.DataGen.python.common import periodic_wavenumbers

        _, _, ksq = periodic_wavenumbers(16)
        self.assertAlmostEqual(float(ksq[1, 0]), float((2.0 * np.pi) ** 2), places=10)
        alpha, t = 1.0e-3, 1.0
        decay = np.exp(-alpha * ksq[1, 0] * t)
        self.assertAlmostEqual(float(decay), float(np.exp(-alpha * (2.0 * np.pi) ** 2)), places=10)

    def test_steady_heat_conduction_schema_and_boundaries(self) -> None:
        sys.path.insert(0, str(ROOT))
        from data.load import PDEloader

        path = self.out_root / "steady_heat_conduction" / "steady_heat_conduction_4-16-16_1.h5"
        with h5py.File(path, "r") as h5:
            for key in ["input_data", "output_data", "u_D", "residual_norm", "picard_iters", "converged"]:
                self.assertIn(key, h5)
            u = h5["output_data"][:, 0]
            u_d = h5["u_D"][:]
            self.assertLess(float(np.max(np.abs(u[:, 0, :] - u_d[:, None]))), 1e-4)
            self.assertLess(float(np.max(np.abs(u[:, :, 0] - u[:, :, 1]))), 1e-4)
            self.assertLess(float(np.max(np.abs(u[:, :, -1] - u[:, :, -2]))), 1e-4)
            self.assertLess(float(np.max(np.abs(u[:, -1, :] - u[:, -2, :]))), 1e-4)
            lam = 1.0 + 0.05 * (u - 298.0)
            self.assertGreater(float(np.min(lam)), 0.0)
            self.assertTrue(np.isfinite(h5["residual_norm"][:]).all())
            self.assertTrue(np.all((h5["converged"][:] == 0) | (h5["converged"][:] == 1)))
        data, label, metadata = PDEloader("steady_heat_conduction").load_data(
            str(self.out_root) + "/",
            size=2,
            return_metadata=True,
        )
        self.assertEqual(tuple(data.shape), (8, 2, 16, 16))
        self.assertEqual(float(label[0]), 10.0)
        self.assertEqual(metadata["channel_names"], ["f", "u"])
        self.assertIn("u_D", metadata["pde_params"])
        self.assertIn("residual_norm", metadata["pde_params"])

    def test_no_leakage_json_and_hashes(self) -> None:
        root_check = self.out_root / "no_leakage_check.json"
        self.assertTrue(root_check.exists())
        root = json.loads(root_check.read_text(encoding="utf-8"))
        self.assertIn("steady_heat_conduction", root["no_leakage_checks"])
        for pde in ["heat", "wave", "advection_diffusion", "steady_heat_conduction"]:
            check_path = self.out_root / pde / "no_leakage_check.json"
            self.assertTrue(check_path.exists())
            check = json.loads(check_path.read_text(encoding="utf-8"))
            self.assertTrue(check["ok"], check)
            self.assertEqual(check["seed_overlap_count"], 0)
            self.assertEqual(check["input_hash_overlap_count"], 0)

    def test_compileall_data(self) -> None:
        subprocess.run([sys.executable, "-m", "compileall", "-q", "data"], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
