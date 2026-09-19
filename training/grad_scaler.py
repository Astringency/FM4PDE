# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch

from torch import Tensor


def get_grad_norm_(parameters, norm_type: float = 2.0) -> Tensor:
    if isinstance(parameters, Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return Tensor(0.0)
    device = parameters[0].grad.device
    if norm_type == torch.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
            ),
            norm_type,
        )
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.amp.GradScaler("cuda")
        self.optimizer_step_succeeded = False

    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True,
    ):
        self.optimizer_step_succeeded = False
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            scale_before = self._scaler.get_scale()
            if not math.isfinite(scale_before) or scale_before <= 0:
                raise FloatingPointError('AMP scale must remain positive and finite')
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                # GradScaler.step() performs the required unscale/non-finite check.
                # Avoid calculating a full-model gradient norm when the caller did
                # not request clipping and does not consume the returned norm.
                norm = None
            self._scaler.step(optimizer)
            self._scaler.update()
            # This wrapper has one optimizer and never supplies a manual scale.
            # A skipped update reduces the scale; successful updates retain or
            # increase it. This also covers fused AdamW's internal skip path.
            self.optimizer_step_succeeded = self._scaler.get_scale() >= scale_before
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
