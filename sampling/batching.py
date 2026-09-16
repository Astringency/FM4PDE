"""Batch cached physical states for repeated-inference experiments."""
import copy

def combine_truths(truths, ids, device):
    import torch
    from sampling.data import PDEGroundTruth
    items = [truths[i] for i in ids]
    def combine(values):
        first = values[0]
        if isinstance(first, torch.Tensor):
            return (torch.cat(values, dim=0) if first.ndim and first.shape[0] == 1 else first.clone()).to(device)
        if isinstance(first, dict):
            return {k: combine([v[k] for v in values]) for k in first}
        if any(v != first for v in values[1:]):
            raise ValueError("Cannot batch differing non-tensor PDE parameters")
        return copy.deepcopy(first)
    coef, sol = torch.cat([x.coef for x in items]).to(device), torch.cat([x.sol for x in items]).to(device)
    meta = copy.deepcopy(items[0].metadata)
    meta.update(sample_ids=[str(i) for i in ids], sample_offsets=ids, batch_size=len(ids), offset=ids[0])
    return PDEGroundTruth(items[0].pde, coef, sol, coef if items[0].pde == "burger" else torch.cat([coef, sol], 1),
                          combine([x.pde_params for x in items]), items[0].channel_names_coef,
                          items[0].channel_names_sol, meta)

