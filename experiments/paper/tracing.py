"""Sampling diagnostics shared by the public timing and error-trajectory runners.

Metric callbacks do not consume random numbers or change sampler tensors.
"""
from __future__ import annotations

import ast
import copy
import time


def diffusion_functions(source, torch, np):
    """Add one callback after the first denoiser call; all original AST stays."""
    tree = ast.parse(source)
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict"
    )
    loop = next(n for n in fn.body if isinstance(n, ast.For))
    pos = next(
        i
        for i, n in enumerate(loop.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "x_N" for t in n.targets)
    )
    footer_start = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "x_final" for t in n.targets)
    )
    decoder = ast.parse("def decode(x_next):\n pass").body[0]
    decoder.body = copy.deepcopy(fn.body[footer_start:])
    # The decoder is exactly the frozen final physical-unit conversion.
    scope = {"torch": torch, "np": np, "F": torch.nn.functional}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[decoder], type_ignores=[])),
            "[physical_decode]",
            "exec",
        ),
        scope,
    )
    original = copy.deepcopy(tree)
    exec(compile(original, "[frozen_diffusion]", "exec"), scope)
    plain = scope["predict"]
    fn.args.args.append(ast.arg(arg="trace"))
    loop.body.insert(
        pos + 1,
        ast.parse("trace(i, decode(x_N), decode(x_cur), float(sigma_t))").body[0],
    )
    ast.fix_missing_locations(tree)
    instrumented = ast.unparse(tree) + "\n"
    exec(compile(tree, "[traced_diffusion]", "exec"), scope)
    return plain, scope["predict"], instrumented


class Trace:
    def __init__(self, cfg, truth, masks, r, normalizer, torch):
        self.cfg, self.truth, self.masks, self.r, self.normalizer, self.torch = (
            cfg,
            truth,
            masks,
            r,
            normalizer,
            torch,
        )
        self.rows = []
        self.paused = 0.0
        self.start = None
        self.metric_cfg = copy.deepcopy(cfg)
        self.metric_cfg.guidance_components = "obs_pde"
        self.metric_cfg.zeta_pde = 1.0
        self.residual_meta = None

    def begin(self):
        self.torch.cuda.synchronize()
        self.start = time.perf_counter()

    def metrics(self, pair):
        torch = self.torch
        a, u = (v.detach() for v in pair)

        # Relative errors are independently accumulated in float64 physical units.
        def rel(p, q, m=None):
            p, q = p.double(), q.double()
            if m is not None:
                p = p * m
                q = q * m
            den = q.norm()
            return float((p - q).norm() / den) if float(den) > 0 else None

        values = {
            "relative_l2_a": (
                None if self.cfg.pde == "burger" else rel(a, self.truth.coef)
            ),
            "relative_l2_u": rel(u, self.truth.sol),
            "observed_relative_l2_a": (
                None
                if self.cfg.pde == "burger"
                else rel(a, self.truth.coef, self.masks.coef)
            ),
            "observed_relative_l2_u": rel(u, self.truth.sol, self.masks.sol),
        }
        state = self.r.SplitState(a, u) if hasattr(self.r, "SplitState") else None
        if state is None:
            from sampling.state import SplitState

            state = SplitState(a, u)
        loss = self.r.compute_guidance_losses(
            state, self.truth, self.masks, self.metric_cfg
        )
        values["L_pde"] = None if loss.L_pde is None else float(loss.L_pde)
        if self.residual_meta is None:
            self.residual_meta = loss.metadata
        for k, v in values.items():
            if v is not None and not __import__("math").isfinite(v):
                raise FloatingPointError(f"nonfinite {self.cfg.pde} {k}: {v}")
        return values

    def __call__(self, step, estimate, native, native_time):
        torch = self.torch
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        active = t0 - self.start - self.paused
        with torch.no_grad():
            row = {
                "step": step,
                "state": (
                    "final_sample"
                    if step == self.cfg.num_steps
                    else "clean_endpoint_estimate"
                ),
                "native_time": native_time,
                "sampling_seconds_excluding_diagnostics": active,
                "elapsed_seconds_including_diagnostics": t0 - self.start,
                **self.metrics(estimate),
            }
            if native is not None:
                row.update({"native_" + k: v for k, v in self.metrics(native).items()})
            else:
                row.update(
                    {"native_" + k: v for k, v in self.metrics(estimate).items()}
                )
        torch.cuda.synchronize()
        self.paused += time.perf_counter() - t0
        row["cumulative_diagnostic_seconds"] = self.paused
        self.rows.append(row)

    def final(self, pair):
        self(
            self.cfg.num_steps,
            pair,
            None,
            0.0 if self.method == "DiffusionPDE" else 1.0,
        )
