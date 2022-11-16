import torch
import copy


# Used in SAC algos
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if len(param.shape) > 1:
            update_target_networks(param, target_param)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def update_target_networks(param, target_param):
    current_density = (param != 0).sum()
    target_density = (target_param != 0).sum()  # torch.count_nonzero(target_param.data)
    difference = target_density - current_density
    # constrain the sparsity by removing the extra elements (smallest values)
    if difference > 0:
        count_rmv = difference
        tmp = copy.deepcopy(abs(target_param.data))
        tmp[tmp == 0] = 10000000
        # rmv_indicies = torch.dstack(unravel_index(torch.argsort(tmp.ravel()), tmp.shape))
        unraveled = unravel_index(torch.argsort(tmp.view(1, -1)[0]), tmp.shape)
        rmv_indicies = torch.stack(unraveled, dim=1)
        rmv_values_smaller_than = tmp[rmv_indicies[count_rmv][0], rmv_indicies[count_rmv][1]]
        target_param.data[tmp < rmv_values_smaller_than] = 0


def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

