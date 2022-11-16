import torch
import torch.nn as nn


def srelu_func(x: torch.Tensor,
               threshold_right,
               slope_right,
               threshold_left,
               slope_left
               ) -> torch.Tensor:
    far_positives = (x > threshold_right)
    far_negatives = (x < threshold_left)
    # middle_ones = torch.logical_not(torch.logical_or(far_negatives, far_positives))
    # not needed, as middle ones keep the same value in output
    output = x.clone()
    output[far_positives] = threshold_right + slope_right * (x[far_positives] - threshold_right)
    output[far_negatives] = threshold_left + slope_left * (x[far_negatives] - threshold_left)
    return output


def srelu_func_per_neuron(x: torch.Tensor,
                          threshold_right,
                          slope_right,
                          threshold_left,
                          slope_left
                          ) -> torch.Tensor:
    """ The four params here are of type nn.Parameter
    with same shape as input x in last dimension (shape[-1]) """
    far_positives = (x > threshold_right)
    far_negatives = (x < threshold_left)
    # middle_ones = torch.logical_not(torch.logical_or(far_negatives, far_positives))
    # not needed, as middle ones keep the same value in output
    output = x.clone()
    # expanding parameters threshold_right (and others) to have same (batch) size
    output[far_positives] = threshold_right.expand_as(x)[far_positives] + slope_right.expand_as(x)[far_positives] * (
                x[far_positives] - threshold_right.expand_as(x)[far_positives])
    output[far_negatives] = threshold_left.expand_as(x)[far_negatives] + slope_left.expand_as(x)[far_negatives] * (
                x[far_negatives] - threshold_left.expand_as(x)[far_negatives])
    return output


def lex_func(x: torch.Tensor,
             multiplier_right,
             exponent_right,
             multiplier_left,
             exponent_left,
             ) -> torch.Tensor:
    positives = (x >= 0)
    negatives = (x < 0)
    output = torch.zeros_like(x)
    output[positives] = multiplier_right * (x[positives] ** exponent_right)
    output[negatives] = -multiplier_left * ((-x[negatives]) ** exponent_left)
    return output


def lex_func_per_neuron(x: torch.Tensor,
                        multiplier_right,
                        exponent_right,
                        multiplier_left,
                        exponent_left,
                        ) -> torch.Tensor:
    """ The four params here are of type nn.Parameter
    with same shape as input x in last dimension (shape[-1]) """
    positives = (x >= 0)
    negatives = (x < 0)
    output = torch.zeros_like(x)
    output[positives] = multiplier_right[positives] * (x[positives] ** exponent_right[positives])
    output[negatives] = -multiplier_left[negatives] * ((-x[negatives]) ** exponent_left[negatives])
    return output


class SymSqrt(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positives = (x >= 0)
        negatives = (x < 0)
        output = torch.zeros_like(x)
        output[positives] = torch.sqrt(x[positives])
        output[negatives] = -torch.sqrt(-x[negatives])
        return output


class SymSqrt1(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        far_positives = (x > 1)
        far_negatives = (x < -1)
        # middle_ones = torch.logical_not(torch.logical_or(far_negatives, far_positives))
        # not needed, as middle ones keep the same value in output
        output = x.clone()
        output[far_positives] = torch.sqrt(x[far_positives])
        output[far_negatives] = -torch.sqrt(-x[far_negatives])
        return output


class NonLEx(nn.Module):
    """ Non-Learnable Exponents (fixed params version of LEx) """

    def __init__(self,
                 multiplier_right: float = 1.,
                 exponent_right: float = 1.,
                 multiplier_left: float = 1.,
                 exponent_left: float = 0.5
                 ):
        super().__init__()
        self.mr = multiplier_right
        self.er = exponent_right
        self.ml = multiplier_left
        self.el = exponent_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return lex_func(x, self.mr, self.er, self.ml, self.el)


class LEx(nn.Module):
    """ Learnable Exponents.
    num_neurons (int): 1 means per-layer shared LEx, >1 means per-neuron LEx.
    Although it takes an int as input, there are only two values legitimate:
    1, or the number of channels(neurons) at input. Default: 1"""

    def __init__(self,
                 multiplier_right: float = 1.,
                 exponent_right: float = 1.,
                 multiplier_left: float = 1.,
                 exponent_left: float = 0.5,
                 num_neurons: int = 1,
                 random_init=False  # this is a (optional) feature, to add later maybe
                 ):
        super().__init__()
        self.num_neurons = num_neurons
        param_shape = (num_neurons,)
        self.mr = nn.Parameter(torch.full(param_shape, float(multiplier_right)))
        self.er = nn.Parameter(torch.full(param_shape, float(exponent_right)))
        self.ml = nn.Parameter(torch.full(param_shape, float(multiplier_left)))
        self.el = nn.Parameter(torch.full(param_shape, float(exponent_left)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_neurons == 1:
            return lex_func(x, self.mr, self.er, self.ml, self.el)
        else:
            return lex_func_per_neuron(x, self.mr, self.er, self.ml, self.el)


class AlternatedLeftReLU(nn.Module):
    """ ALL-ReLU activation function
    The alternating behavior of this function needs to be implemented by yourself at a higher level of abstraction
    (by giving each layer -alpha or alpha as the slope_left)
    Function that is then implemented is:
                   -alpha * x    if x < 0 and layer_index % 2 == 0
        f(x) = {    alpha * x    if x < 0 and layer_index % 2 == 1
                        x        if x > 0
    The input layer (layer_index = 1) and output layer (layer_index = L) are excluded.
    See the paper: Truly Sparse Neural Networks at Scale, by Curci, Mocanu & Pechenizkiy:
    https://arxiv.org/abs/2102.01732
    """
    def __init__(self, slope_left: float):
        super().__init__()
        self.slope_left = slope_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        negatives = (x < 0)
        output = x.clone()
        output[negatives] = self.slope_left * x[negatives]
        return output


class FixedSReLU(nn.Module):
    def __init__(self,
                 threshold_right: float = 0.4,
                 slope_right: float = 0.2,
                 threshold_left: float = -0.4,
                 slope_left: float = 0.2
                 ):
        super().__init__()
        self.tr = threshold_right
        self.ar = slope_right
        self.tl = threshold_left
        self.al = slope_left

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return srelu_func(x, self.tr, self.ar, self.tl, self.al)


class SReLU(nn.Module):
    """ Activation function SReLU.
    num_neurons (int): 1 means per-layer shared SReLU, >1 means per-neuron SReLU.
    Although it takes an int as input, there are only two values legitimate:
    1, or the number of channels(neurons) at input. Default: 1
    """

    def __init__(self,
                 threshold_right: float = 0.4,
                 slope_right: float = 0.2,
                 threshold_left: float = -0.4,
                 slope_left: float = 0.2,
                 num_neurons: int = 1,
                 random_init=False  # this is a new (optional) feature, by Bram
                 ):
        super().__init__()
        self.num_neurons = num_neurons
        param_shape = (num_neurons,)
        self.tr = nn.Parameter(torch.full(param_shape, float(threshold_right)))
        self.ar = nn.Parameter(torch.full(param_shape, float(slope_right)))
        self.tl = nn.Parameter(torch.full(param_shape, float(threshold_left)))
        self.al = nn.Parameter(torch.full(param_shape, float(slope_left)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_neurons == 1:
            return srelu_func(x, self.tr, self.ar, self.tl, self.al)
        else:
            return srelu_func_per_neuron(x, self.tr, self.ar, self.tl, self.al)


def setup_act_func_args(act_func_args, activation, num_hid_neurons):
    """ Sets up the arguments for the activation function.
    :param act_func_args: list of floats
    :param activation: str, name of activation function
    :param num_hid_neurons: int, for example 256
    :return: act_args: a list of floats (parameters for the activation func)
             per_neuron: a dict with the number of neurons to share params with (1 or num_hid_neurons)
    """
    act_args, per_neuron = [], {}
    act_funcs_with_per_neuron_option = ['srelu', 'lex']
    act_funcs_with_params = ['srelu', 'fixedsrelu', 'elu', 'leakyrelu', 'nonlex', 'lex']
    if activation in act_funcs_with_per_neuron_option and act_func_args[1]:
        per_neuron = {"num_neurons": num_hid_neurons}
    if act_func_args[0] is not None and activation in act_funcs_with_params:
        for arg in act_func_args[0]:
            act_args.append(arg)
    return act_args, per_neuron


act_funcs = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'elu': nn.ELU, 'leakyrelu': nn.LeakyReLU,
             'symsqrt': SymSqrt, 'symsqrt1': SymSqrt1, 'nonlex': NonLEx, 'lex': LEx,
             'fixedsrelu': FixedSReLU, 'srelu': SReLU, 'allrelu': AlternatedLeftReLU,
             'swish': nn.SiLU, 'selu': nn.SELU,
             }


def setup_activation_funcs_list(activation, act_func_args, num_hid_layers, num_hid_neurons):
    activation_funcs = nn.ModuleList()
    if activation == 'allrelu':
        for hid_layer in range(num_hid_layers):
            if hid_layer % 2 == 0:
                activation_funcs.append(act_funcs['allrelu'](-act_func_args[0][0]))
            else:
                activation_funcs.append(act_funcs['allrelu'](act_func_args[0][0]))
    else:
        act_args, per_neuron = setup_act_func_args(act_func_args, activation, num_hid_neurons)
        for hid_layer in range(num_hid_layers):
            activation_funcs.append(act_funcs[activation](*act_args, **per_neuron))
    return activation_funcs


if __name__ == '__main__':
    # to test whether gradients are computed correctly

    inpt = [[4., -0.4, 1.5, -100],
            [3., -1, 1.2, -30],
            [2, 5.3, 0.4, 0.01]]

    # act_func = SymSqrt()
    # act_func = SymSqrt1()
    # act_func = FixedSReLU(threshold_right=0.4, slope_right=0.2, threshold_left=-0.4, slope_left=0.2)
    # act_func = SReLU(threshold_right=0.4, slope_right=0.2, threshold_left=-0.4, slope_left=0.2, num_neurons=len(inpt[0]))
    # act_func = NonLEx(multiplier_right=1, exponent_right=1, multiplier_left=1, exponent_left=0.5)
    # act_func = LEx(multiplier_right=1, exponent_right=1, multiplier_left=1, exponent_left=0.5, num_neurons=len(inpt[0]))
    act_func = AlternatedLeftReLU(slope_left=-0.6)

    x = torch.tensor(inpt, requires_grad=True)
    out = act_func.forward(x)
    loss = torch.sum(out)
    loss.backward()

    print(x.grad)

    # for SReLU
    # print(act_func.tr.grad)  # 1 - ar
    # print(act_func.ar.grad)  # x - tr
    # print(act_func.tl.grad)  # 1 - al
    # print(act_func.al.grad)  # x - tl

    # for LEx
    # print(act_func.mr.grad)
    # print(act_func.er.grad)
    # print(act_func.ml.grad)
    # print(act_func.el.grad)
