import torch
import numpy as np

def to_gym_action(action_c, action_d, flat_actions=True):
    # assuming both are torch tensors
    if flat_actions:
        ac = action_c.tolist()[0]
    else:
        ac = action_c.unsqueeze(-1).tolist()[0]
    ad = action_d.squeeze().item()
    return [ad, ac]

def gym_to_buffer(action, flat_actions=True):
    ad = action[0]
    if flat_actions:
        ac = np.hstack(action[1:])
    else:
        ac = action[1]
    return [ad] + np.array(ac).flatten().tolist()

def to_torch_action(actions, device):
    ad = torch.Tensor(actions[:, 0]).int().to(device)
    ac = torch.Tensor(actions[:, 1:]).to(device)
    return ac, ad
