import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor
from typing import List


class MaskAdam(Optimizer):
    r"""Implements the MaskAdam optimizer.
    Especially useful for Dynamic Sparse Training.
    The difference with regular Adam is that the gradients and its first and
    second (raw) moments are masked for non-existing connections.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, masks, closure=None):
        """Performs a single optimization step.
        Args:
            masks (list): List of masks (torch tensors) for each layer in the network.
                          Should be of length equal to the number of connection-layers in the network,
                          and should have element None if a layer is dense.
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            params = group['params']
            beta1, beta2 = group['betas']
            amsgrad = group['amsgrad']

            for p in params:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('MaskAdam does not support sparse gradients, '
                                           'which is something distinct from sparse connectivity. '
                                           'Please consider SparseAdam instead if you have sparse gradients.')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = torch.tensor(0.)
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    state['step'].add_(1)

                    # record the updates
                    state_steps.append(state['step'])
                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            self.mask_adam(params_with_grad,
                           grads,
                           exp_avgs,
                           exp_avg_sqs,
                           max_exp_avg_sqs,
                           state_steps,
                           masks,
                           amsgrad=amsgrad,
                           beta1=beta1,
                           beta2=beta2,
                           lr=group['lr'],
                           weight_decay=group['weight_decay'],
                           eps=group['eps'],
                           )
        return loss

    def mask_adam(self,
                  params: List[Tensor],
                  grads: List[Tensor],
                  exp_avgs: List[Tensor],
                  exp_avg_sqs: List[Tensor],
                  max_exp_avg_sqs: List[Tensor],
                  state_steps: List[Tensor],
                  masks: List[Tensor or None],
                  *,
                  amsgrad: bool,
                  beta1: float,
                  beta2: float,
                  lr: float,
                  weight_decay: float,
                  eps: float,
                  ):
        """
        Function that performs the MaskAdam optimizer computation.
        It uses masks to simulate sparsity. Some layers may not have masks.

        ASSUMES: all connection-layers (weight matrices) have gradients
        (they are in params_with_grad in step function above, and so are given in param argument here)
        """
        layer_idx = 0
        for param_idx, param in enumerate(params):
            grad = grads[param_idx]
            exp_avg = exp_avgs[param_idx]
            exp_avg_sq = exp_avg_sqs[param_idx]
            step = state_steps[param_idx]

            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            if len(param.size()) == 2:
                mask = masks[layer_idx]
                layer_idx += 1
                if mask is not None:
                    # this is a sparse layer, everything is masked (put to zero) for non-existing connections
                    grad[mask == 0] = 0
                    exp_avg[mask == 0] = 0
                    exp_avg_sq[mask == 0] = 0
                    if amsgrad:
                        max_exp_avg_sqs[param_idx][mask == 0] = 0

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.max(max_exp_avg_sqs[param_idx], exp_avg_sq, out=max_exp_avg_sqs[param_idx])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[param_idx] / bias_correction2).sqrt().add_(eps)
            else:
                denom = (exp_avg_sq / bias_correction2).sqrt().add_(eps)

            step_size = lr / bias_correction1
            param -= step_size * exp_avg / denom
