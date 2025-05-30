import torch
import torch.nn.functional as F

def Loss(output1, output2, l2_weight=0.1, cosine_weight=1.0, dim=-1, eps=1e-8, **kwargs):
    """
    Combines L2 loss (scaled) and cosine similarity loss.
    output1: Typically projected_audio_embeddings
    output2: Typically image_embeddings_emu
    kwargs: Additional arguments from argparse, used for l2_weight and cosine_weight.
    """
    l2_loss_val = F.mse_loss(output1, output2)
    
    cosine_loss_val = 1 - F.cosine_similarity(output1, output2, dim=dim, eps=eps).mean()
    
    # Use weights from kwargs if provided, otherwise use defaults
    current_l2_weight = kwargs.get('l2_weight', l2_weight)
    current_cosine_weight = kwargs.get('cosine_weight', cosine_weight)

    total_loss = (current_l2_weight * l2_loss_val) + (current_cosine_weight * cosine_loss_val)
    return total_loss 