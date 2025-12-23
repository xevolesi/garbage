
def get_cpu_state_dict(state_dict):
    return {name: tensor.detach().cpu() for name, tensor in state_dict.items()}