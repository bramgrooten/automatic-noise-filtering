import torch
import torch.nn as nn
import copy
from utils.activations import act_funcs, setup_act_func_args


class SparseBaseAgent:
    """ Sparse Base Agent class to inherit """
    def __init__(self):
        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None
        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None
        self.device = None
        self.total_it = None
        self.prev_permutations = None

    def save(self, filename):
        checkpoint = {
            "iteration": self.total_it,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_masks": self.actor.masks,
            "actor_torch_masks": self.actor.torch_masks,
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "critic_q1_masks": self.critic.q1_masks,
            "critic_q2_masks": self.critic.q2_masks,
            "critic_q1_torch_masks": self.critic.q1_torch_masks,
            "critic_q2_torch_masks": self.critic.q2_torch_masks,
            "prev_permutations": self.prev_permutations,
        }
        # torch.save(checkpoint, f"{filename}_iter_{self.total_it}")
        torch.save(checkpoint, filename)
        print(f"Saved current model in: {filename}")

    def load(self, filename, load_device=None):
        if load_device is None:
            load_device = self.device
        loaded_checkpoint = torch.load(filename, map_location=load_device)
        self.total_it = loaded_checkpoint["iteration"]
        self.actor.load_state_dict(loaded_checkpoint["actor"])
        self.actor_target.load_state_dict(loaded_checkpoint["actor_target"])
        self.actor_optimizer.load_state_dict(loaded_checkpoint["actor_optimizer"])
        self.actor.masks = loaded_checkpoint["actor_masks"]
        self.actor.torch_masks = loaded_checkpoint["actor_torch_masks"]
        self.critic.load_state_dict(loaded_checkpoint["critic"])
        self.critic_target.load_state_dict(loaded_checkpoint["critic_target"])
        self.critic_optimizer.load_state_dict(loaded_checkpoint["critic_optimizer"])
        self.critic.q1_masks = loaded_checkpoint["critic_q1_masks"]
        self.critic.q2_masks = loaded_checkpoint["critic_q2_masks"]
        self.critic.q1_torch_masks = loaded_checkpoint["critic_q1_torch_masks"]
        self.critic.q2_torch_masks = loaded_checkpoint["critic_q2_torch_masks"]
        self.prev_permutations = loaded_checkpoint.get("prev_permutations")
        print(f"Loaded model from: {filename}")


class BaseAgent:
    """ Base Agent class to inherit """
    def __init__(self):
        self.actor = None
        self.actor_target = None
        self.actor_optimizer = None
        self.critic = None
        self.critic_target = None
        self.critic_optimizer = None
        self.device = None
        self.total_it = None
        self.prev_permutations = None

    def save(self, filename):
        checkpoint = {
            "iteration": self.total_it,
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "prev_permutations": self.prev_permutations,
        }
        # torch.save(checkpoint, f"{filename}_iter_{self.total_it}")
        torch.save(checkpoint, filename)
        print(f"Saved current model in: {filename}")

    def load(self, filename, load_device=None):
        if load_device is None:
            load_device = self.device
        loaded_checkpoint = torch.load(filename, map_location=load_device)
        self.total_it = loaded_checkpoint["iteration"]
        self.actor.load_state_dict(loaded_checkpoint["actor"])
        self.actor_target.load_state_dict(loaded_checkpoint["actor_target"])
        self.actor_optimizer.load_state_dict(loaded_checkpoint["actor_optimizer"])
        self.critic.load_state_dict(loaded_checkpoint["critic"])
        self.critic_target.load_state_dict(loaded_checkpoint["critic_target"])
        self.critic_optimizer.load_state_dict(loaded_checkpoint["critic_optimizer"])
        self.prev_permutations = loaded_checkpoint.get("prev_permutations")
        print(f"Loaded model from: {filename}")


class Agent:
    """ Old version of the base agent. Didn't save the target networks. """
    def __init__(self):
        self.actor = None
        self.critic = None
        self.actor_target = None
        self.critic_target = None
        self.actor_optimizer = None
        self.critic_optimizer = None
        self.device = None
        self.total_it = None

    def save(self, filename):
        checkpoint = {
            "iteration": self.total_it,
            "critic": self.critic.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
        }
        # torch.save(checkpoint, f"{filename}_iter_{self.total_it}")
        torch.save(checkpoint, filename)
        print(f"Saved current model in: {filename}")

    def load(self, filename, load_device=None):
        if load_device is None:
            load_device = self.device
        loaded_checkpoint = torch.load(filename, map_location=load_device)
        self.total_it = loaded_checkpoint["iteration"]
        self.critic.load_state_dict(loaded_checkpoint["critic"])
        self.critic_optimizer.load_state_dict(loaded_checkpoint["critic_optimizer"])
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(loaded_checkpoint["actor"])
        self.actor_optimizer.load_state_dict(loaded_checkpoint["actor_optimizer"])
        self.actor_target = copy.deepcopy(self.actor)
        print(f"Loaded model from: {filename}")




