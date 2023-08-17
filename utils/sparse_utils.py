import numpy as np
import torch


def initialize_mask(layer_name, layer_sparsity_level, num_in_neurons, num_out_neurons, print_info=True):
    """ New version of initializeSparsityLevelWeightMask
    that produces the exact sparsity level, instead of something closeby. """
    total_connections = num_in_neurons * num_out_neurons
    num_connections = int((1 - layer_sparsity_level) * total_connections)
    mask = np.zeros((num_in_neurons, num_out_neurons))
    mask = grow_set(mask, num_connections)
    # if print_info:
    #     print(f"Sparsity Level Initialization {layer_name}: sparsity {1 - np.sum(mask)/total_connections:.6f}; "
    #           f"num active connections {num_connections}; num_in_neurons {num_in_neurons}; "
    #           f"num_out_neurons {num_out_neurons}; num_dense_connections {total_connections}")
    #     print(f" OutDegreeBottomNeurons {np.mean(mask.sum(axis=1)):.2f} ± {np.std(mask.sum(axis=1)):.2f};"
    #           f" InDegreeTopNeurons {np.mean(mask.sum(axis=0)):.2f} ± {np.std(mask.sum(axis=0)):.2f}")
    return num_connections, mask.transpose()


def adjust_connectivity_set(weights, num_weights, zeta, mask):
    """ New version of changeConnectivitySET that uses a slightly different method:
     remove the zeta fraction of weights that are nearest to zero (so zeta smallest absolute values,
     instead of zeta largest negative and zeta smallest positive).
    """
    # mask = prune_set(weights, zeta)
    mask = prune_set_only_active(weights, zeta, mask, num_weights)
    mask = grow_set(mask, num_weights)
    return mask


def prune_set(weights, zeta):
    """ Remove zeta fraction of weights that are nearest to zero. """
    abs_weights = np.abs(weights)
    abs_vector = np.sort(abs_weights.ravel())
    first_nonzero = np.nonzero(abs_vector)[0][0]
    threshold = abs_vector[int(first_nonzero + (len(abs_vector) - first_nonzero) * zeta)]
    # wrong: there could also be active connections that still have a weight of 0.
    # Hmm, this should be fixed by grow_set though (more connections than zeta will grow back in that case)
    new_mask = np.zeros_like(weights)
    new_mask[abs_weights > threshold] = 1
    return new_mask


def prune_set_only_active(weights, zeta, mask, num_weights):
    """ Remove zeta fraction of weights that are nearest to zero.
    Prune threshold now also based on existing connections that have value 0. """
    active = np.where(mask == 1)
    abs_weights = np.abs(weights)

    abs_vector = np.sort(abs_weights[active].ravel())
    assert len(abs_vector) == num_weights
    threshold = abs_vector[int(num_weights * zeta)]
    # assumes threshold is unique, otherwise more connections than zeta might be pruned. They will grow back.
    # prune_set assumes this as well.

    new_mask = np.zeros_like(weights)
    new_mask[abs_weights > threshold] = 1
    return new_mask


def grow_set(mask, num_weights):
    """ Grow new connections according to SET: Choose randomly from the available options. """
    num_to_grow = num_weights - np.sum(mask)
    idx_zeros_i, idx_zeros_j = np.where(mask == 0)
    new_conn_idx = np.random.choice(idx_zeros_i.shape[0], int(num_to_grow), replace=False)
    mask[idx_zeros_i[new_conn_idx], idx_zeros_j[new_conn_idx]] = 1
    return mask


def critic_give_new_connections_init_values(critic, q1_old_masks, q2_old_masks, init_new_weights_method, device):
    for layer in range(critic.num_hid_layers + 1):
        if layer == 0:
            layer_give_new_connections_init_values(
                critic.q1_input_layer.weight, critic.q1_masks[0], q1_old_masks[0], init_new_weights_method, device)
            layer_give_new_connections_init_values(
                critic.q2_input_layer.weight, critic.q2_masks[0], q2_old_masks[0], init_new_weights_method, device)
        elif layer == critic.num_hid_layers:
            layer_give_new_connections_init_values(
                critic.q1_output_layer.weight, critic.q1_masks[layer], q1_old_masks[layer], init_new_weights_method, device)
            layer_give_new_connections_init_values(
                critic.q2_output_layer.weight, critic.q2_masks[layer], q2_old_masks[layer], init_new_weights_method, device)
        else:
            layer_give_new_connections_init_values(
                critic.q1_hid_layers[layer-1].weight, critic.q1_masks[layer], q1_old_masks[layer], init_new_weights_method, device)
            layer_give_new_connections_init_values(
                critic.q2_hid_layers[layer-1].weight, critic.q2_masks[layer], q2_old_masks[layer], init_new_weights_method, device)


def actor_give_new_connections_init_values(actor, old_masks, init_new_weights_method, device):
    for layer in range(actor.num_hid_layers + 1):
        if layer == 0:
            layer_give_new_connections_init_values(
                actor.input_layer.weight, actor.masks[0], old_masks[0], init_new_weights_method, device)
        elif layer == actor.num_hid_layers:
            layer_give_new_connections_init_values(
                actor.output_layer.weight, actor.masks[layer], old_masks[layer], init_new_weights_method, device)
        else:
            layer_give_new_connections_init_values(
                actor.hid_layers[layer-1].weight, actor.masks[layer], old_masks[layer], init_new_weights_method, device)


def layer_give_new_connections_init_values(layer_weights, new_mask, old_mask, init_new_weights_method, device):
    if init_new_weights_method == "unif":
        reinit_values_unif(layer_weights, new_mask, old_mask, device)
    elif init_new_weights_method == "xavier":
        reinit_values_xavier(layer_weights, new_mask, old_mask, device)
    else:
        raise ValueError("Unknown init_new_weights_method")


def reinit_values_unif(layer_weights, new_mask, old_mask, device):
    # the default initialization values of PyTorch
    # see https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
    if old_mask is not None:
        weights = layer_weights.data.cpu().numpy()
        diff = new_mask - old_mask  # new connections will have value 1 in diff
        num_in_neurons = layer_weights.data.shape[1]  # weight matrix is transposed
        bound = 1 / np.sqrt(num_in_neurons)
        weights[diff == 1] = np.random.uniform(-bound, bound)  # only new connections will get new values
        layer_weights.data = torch.from_numpy(weights).float().to(device)


def reinit_values_xavier(layer_weights, new_mask, old_mask, device):
    # also called Glorot initialization
    # see https://keras.io/api/layers/initializers/#glorotuniform-class
    # and https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    if old_mask is not None:
        weights = layer_weights.data.cpu().numpy()
        diff = new_mask - old_mask  # new connections will have value 1 in diff
        num_in_neurons = layer_weights.data.shape[1]  # weight matrix is transposed
        num_out_neurons = layer_weights.data.shape[0]
        bound = np.sqrt(12 / (num_in_neurons + num_out_neurons))  # 12 = 2 * 6 (see 'gain' variable in links above)
        weights[diff == 1] = np.random.uniform(-bound, bound)  # only new connections will get new values
        layer_weights.data = torch.from_numpy(weights).float().to(device)


def print_sparsities(critic_params, critic_target_params, actor_params, actor_target_params=None):
    sparsities_dict = {}
    sparsities_dict['critic_sparsity'], layers_sp = print_sparsity_one_network(critic_params, 'critic')
    for lay_idx, lay_sp in enumerate(layers_sp):
        sparsities_dict[f'critic_sparsity_layer{lay_idx}'] = lay_sp

    sparsities_dict['critic_target_sparsity'], layers_sp = print_sparsity_one_network(critic_target_params, 'critic_target')
    for lay_idx, lay_sp in enumerate(layers_sp):
        sparsities_dict[f'critic_target_sparsity_layer{lay_idx}'] = lay_sp

    sparsities_dict['actor_sparsity'], layers_sp = print_sparsity_one_network(actor_params, 'actor')
    for lay_idx, lay_sp in enumerate(layers_sp):
        sparsities_dict[f'actor_sparsity_layer{lay_idx}'] = lay_sp

    if actor_target_params is not None:
        sparsities_dict['actor_target_sparsity'], layers_sp = print_sparsity_one_network(actor_target_params, 'actor_target')
        for lay_idx, lay_sp in enumerate(layers_sp):
            sparsities_dict[f'actor_target_sparsity_layer{lay_idx}'] = lay_sp
    return sparsities_dict


def print_sparsity_one_network(params, network_name):
    # layer = 0
    total_pruned_connections = 0
    total_connections_possible = 0
    layer_sparsities = []
    for param in params:
        if len(param.shape) > 1:
            num_pruned_connections = np.sum(param.data.cpu().numpy() == 0)
            total_layer_connections = param.shape[0] * param.shape[1]
            this_layer_current_sparsity = num_pruned_connections / total_layer_connections
            # print(f"{network_name} sparsity layer {layer}: {this_layer_current_sparsity}")
            # layer += 1
            layer_sparsities.append(round(this_layer_current_sparsity, 5))
            total_pruned_connections += num_pruned_connections
            total_connections_possible += total_layer_connections
    global_sparsity = total_pruned_connections / total_connections_possible
    # print(f"  {round(global_sparsity, 5)}  {network_name} global sparsity. Layer sparsities: {layer_sparsities}")
    return global_sparsity, layer_sparsities


def compute_sparsity_per_layer(global_sparsity, neuron_layers, keep_dense, closeness=0.2, method='ER', input_layer_sparsity=-1.):
    """
    Function to compute the sparsity levels of individual layers, based on a given global sparsity level.
    Instead of a uniform sparsity (for example 80% in every layer),
    this function gives bigger layers a larger sparsity level.
    Assumes an MLP architecture.
    :param global_sparsity: float, number between 0 and 1, the desired global sparsity of the whole network
    :param neuron_layers: list of ints, number of neurons in each neuron-layer
    :param keep_dense: list of bools, must be of length neuron_layers-1.
                       Put a True if this connection-layer should be dense, and a False if not.
    :param closeness: float, the exponent in the computation, between 0 and 1.
                      Only used in method 'new'. Determines how close the sparsity levels of layers should be.
                      Value closer to 0 gives more uniform (0 = same sparsity level in each sparse layer)
                      Value closer to 1 gives less uniform (1 = most differences in sparsity levels)
    :param method: str, 'new' or 'ER' or 'uniform'. ER is from Mocanu et al. 2018:
                   https://www.nature.com/articles/s41467-018-04316-3 Uniform gives each layer same sparsity level
                   (which needs to be higher than the desired global sparsity if you want some dense layers as well.)
    :param input_layer_sparsity: float, number between 0 and 1, the desired sparsity of the input layer. Default is -1,
                                meaning that the input layer sparsity is computed based on the global sparsity.
    :return: list of floats (between 0 and 1) giving the sparsity of each connection-layer (length: neuron_layers-1)

    Example:
    compute_sparsity_per_layer(global_sparsity=0.8, neuron_layers=[17, 256, 256, 6], keep_dense=[False, False, True])
    output: [0.40, 0.85, 0.0]  (output is normally not rounded, just for brevity here)
    """
    assert len(neuron_layers) - 1 == len(keep_dense)

    total_connections_possible = 0
    connections_possible_per_layer = []
    for n_layer_idx in range(len(neuron_layers)-1):
        layer_connections = neuron_layers[n_layer_idx] * neuron_layers[n_layer_idx + 1]
        total_connections_possible += layer_connections
        connections_possible_per_layer.append(layer_connections)

    global_density = 1 - global_sparsity
    total_connections_needed = round(global_density * total_connections_possible)

    keep_input_layer_fixed = False
    if input_layer_sparsity == 0:
        keep_dense[0] = True
    elif 0 < input_layer_sparsity < 1:
        keep_input_layer_fixed = True
    elif input_layer_sparsity != -1:
        raise ValueError("input_layer_sparsity must be in the interval [0,1) "
                         "or equal to -1 to let it be computed based on global sparsity")

    total_conn_needed_sparse_lays = total_connections_needed
    keep_probs = []  # the probability of keeping a connection, for each layer (i.e. the layer density)
    for c_layer_idx, c_layer_dense in enumerate(keep_dense):
        if c_layer_dense:
            keep_probs.append(1)
            total_conn_needed_sparse_lays -= connections_possible_per_layer[c_layer_idx]
        else:
            if c_layer_idx == 0 and input_layer_sparsity > 0:
                prob = 1 - input_layer_sparsity
                total_conn_needed_sparse_lays -= round(prob * connections_possible_per_layer[c_layer_idx])
            else:
                if method == 'new':
                    prob = 2 / ((neuron_layers[c_layer_idx] * neuron_layers[c_layer_idx + 1]) ** closeness)
                elif method == 'ER':
                    prob = (neuron_layers[c_layer_idx] + neuron_layers[c_layer_idx + 1]) \
                           / (neuron_layers[c_layer_idx] * neuron_layers[c_layer_idx + 1])
                elif method == 'uniform':
                    prob = global_density
                else:
                    raise ValueError('Unknown method name for computing layer sparsities.')
            keep_probs.append(prob)

    # Counting the number of connections that we have in the sparse layers (that don't stay on a fixed sparsity level)
    total_conn_current_sparse_lays = 0
    for c_layer_idx, c_layer_dense in enumerate(keep_dense):
        if not c_layer_dense and not (keep_input_layer_fixed and c_layer_idx == 0):
            num_connections = round(keep_probs[c_layer_idx] * connections_possible_per_layer[c_layer_idx])
            total_conn_current_sparse_lays += num_connections

    if total_conn_current_sparse_lays == 0:
        return handle_impossible_config_input_layer(input_layer_sparsity, keep_probs, connections_possible_per_layer,
                                                    total_connections_possible)

    adjustment_factor = total_conn_needed_sparse_lays / total_conn_current_sparse_lays
    for c_layer_idx, c_layer_dense in enumerate(keep_dense):
        if not c_layer_dense and not (keep_input_layer_fixed and c_layer_idx == 0):
            keep_probs[c_layer_idx] *= adjustment_factor
    # print(f"adjustment_factor (epsilon) is {adjustment_factor}")

    # Check if all probabilities are valid
    do_again = False
    for c_layer_idx, keep_prob in enumerate(keep_probs):
        prob_for_one_connection = 1 / connections_possible_per_layer[c_layer_idx]
        minimum_connections = max(neuron_layers[c_layer_idx], neuron_layers[c_layer_idx+1])
        prob_minimum_connections = minimum_connections / connections_possible_per_layer[c_layer_idx]
        if keep_prob > 1:
            keep_dense[c_layer_idx] = True
            do_again = True
        elif keep_prob < prob_for_one_connection:
            # example input: global_sparsity=0.9, neuron_layers=[10, 256, 256], keep_dense=[False, True]
            # can happen if dense layers & high sparsity is desired
            raise ValueError(f"This sparsity configuration is impossible, empty layers are required (layer {c_layer_idx}).")
        elif keep_prob < prob_minimum_connections:
            # warn if prob so low that this c_layer would have fewer connections than num_neurons on either side
            print(f"\nWARNING: extremely sparse layer, some neurons will have no connections in layer: {c_layer_idx}.\n")
    if do_again:
        return compute_sparsity_per_layer(global_sparsity, neuron_layers, keep_dense, closeness, method, input_layer_sparsity)

    return collect_output_sparsity_per_layer(keep_probs, connections_possible_per_layer, total_connections_possible)


def collect_output_sparsity_per_layer(keep_probs, connections_possible_per_layer, total_connections_possible):
    sparsity_per_layer = []
    total_connections = 0
    for c_layer_idx, keep_prob in enumerate(keep_probs):
        sparsity_per_layer.append(float(1 - keep_prob))
        total_connections += round(keep_prob * connections_possible_per_layer[c_layer_idx])

    print(f"\nconnections: {total_connections}, out of: {total_connections_possible}, "
          f"global sparsity: {round(1 - total_connections/total_connections_possible, 6)}")
    print(f"sparsity per layer: {sparsity_per_layer}")
    return sparsity_per_layer


def handle_impossible_config_input_layer(input_layer_sparsity, keep_probs, connections_possible_per_layer,
                                         total_connections_possible):
    assert 0 < input_layer_sparsity < 1, "other situation not handled yet"
    conns = total_connections_possible - round(input_layer_sparsity * connections_possible_per_layer[0])
    print(f"\nUsing higher global sparsity than requested."
          f"\nYour chosen sparsity configuration is impossible. Minimum sparsity level with "
          f"input_layer_sparsity {input_layer_sparsity} is {1 - conns / total_connections_possible}."
          f"\nConfiguring that now :)")
    new_keep_probs = [1] * len(keep_probs)
    new_keep_probs[0] = 1 - input_layer_sparsity
    return collect_output_sparsity_per_layer(new_keep_probs, connections_possible_per_layer, total_connections_possible)



if __name__ == '__main__':
    # to test some of the functions

    ### to get an overview for each environment
    env_dims = {
                'HalfCheetah-v3': (17, 6),
                # 'Walker2d-v3': (17, 6),
                # 'Hopper-v3': (11, 3),
                'Humanoid-v3': (376, 17),
                # 'Ant-v3': (111, 8),
                # 'Swimmer-v3': (8, 2),
                # 'SlipperyAnt': (29, 8)
                }

    # glob_sparsities = [0, .5, .8, .9, .95, 0.96, 0.97]  # , .98]
    glob_sparsities = [0]
    fake_feats_multiplier = 10

    for env_name, env_dims in env_dims.items():
        print(f"\n{env_name}")
        for glob_sparsity in glob_sparsities:
            for method in ['uniform']:  # 'ER',
                sparsities = compute_sparsity_per_layer(global_sparsity=glob_sparsity,
                                                        # neuron_layers=[env_dims[0]*fake_feats_multiplier, 256, 256, 2 * env_dims[1]],  # SAC has 2 output heads in actor
                                                        # neuron_layers=[env_dims[0]*fake_feats_multiplier, 256, 256, env_dims[1]],  # TD3 has 1 output head in actor.
                                                        neuron_layers=[env_dims[0]*fake_feats_multiplier + env_dims[1], 256, 256, 1],  # for all critic networks
                                                        keep_dense=[False, False, True],
                                                        method=method,
                                                        input_layer_sparsity=0.8)
                print(f"desired global: {glob_sparsity}, sparsity per layer {method[:2]}: {sparsities}")




