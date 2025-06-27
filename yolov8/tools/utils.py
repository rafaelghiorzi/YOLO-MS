import torch
import yaml

def load_config(config_path):
    """Loads a YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_optimizer(model_params, config):
    """Initializes and returns an optimizer based on the config."""
    optimizer_name = config['training']['optimizer'].lower()
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']

    if optimizer_name == 'adam':
        betas = config['training'].get('adam_betas', [0.9, 0.999])
        return torch.optim.Adam(model_params, lr=lr, betas=tuple(betas), weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config['training'].get('sgd_momentum', 0.937)
        nesterov = config['training'].get('sgd_nesterov', True)
        return torch.optim.SGD(model_params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_scheduler(optimizer, config):
    """Initializes and returns a learning rate scheduler based on the config."""
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'none').lower()
    
    if scheduler_type == 'cosine':
        t_max = scheduler_config.get('cosine_t_max', config['training']['epochs'])
        eta_min = scheduler_config.get('cosine_eta_min', 0.00001)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_lr_size', 30)
        gamma = scheduler_config.get('step_lr_gamma', 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'none':
        return None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")