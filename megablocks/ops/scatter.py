import torch
from megablocks.backend import kernels
from stk.backend.autocast import custom_fwd, custom_bwd


# Autograd wrapper for scatter kernel.
class ScatterOp(torch.autograd.Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, x, indices, bin_ids, weights, bins, top_k, tp, tp_group):
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        ctx.tp = tp
        ctx.tp_group = tp_group
        results = kernels.scatter(
            x, indices, bin_ids, weights, bins, top_k)
        return results
            

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins = saved_tensors[:4]

        wgrad, handle = None, None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = kernels.scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                ctx.top_k)
            if ctx.tp:
                handle = torch.distributed.all_reduce(
                    wgrad, 
                    op=torch.distributed.ReduceOp.SUM, 
                    group=ctx.tp_group,
                    async_op=True,
                )
            
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = kernels.gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                ctx.top_k)
        
        if handle is not None:
            handle.wait()
            
        return dgrad, None, None, wgrad, None, None, None, None


def scatter(x: torch.Tensor,
            indices: torch.Tensor,
            bin_ids: torch.Tensor,
            weights: torch.Tensor,
            bins: torch.Tensor,
            top_k: int, 
            tp=False,
            tp_group=None):
    return ScatterOp.apply(x, indices, bin_ids, weights, bins, top_k, tp, tp_group)
