import torch
import yaml
import os

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

def load_pretrained_weights(model, pretrained_path, strict=False):
    """Load pretrained weights with proper error handling."""
    if not pretrained_path or not os.path.exists(pretrained_path):
        print("No pretrained weights found, training from scratch")
        return model
    
    try:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)
        
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
            
        print(f"Successfully loaded pretrained weights from {pretrained_path}")
        
    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        print("Training from scratch")
    
    return model

def freeze_layers(model, freeze_patterns):
    """Freeze specific layers based on name patterns."""
    frozen_count = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in freeze_patterns):
            param.requires_grad = False
            frozen_count += 1
            print(f"Frozen: {name}")
    
    print(f"Frozen {frozen_count} parameters")
    return model