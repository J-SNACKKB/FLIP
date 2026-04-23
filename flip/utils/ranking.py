import torch
from torch.nn import functional as F

def make_square_and_get_triu(input):
    input_by_input = input - input.transpose(0, 1)
    idx = torch.triu_indices(len(input), len(input), 1, device=input.device)
    return input_by_input[idx[0], idx[1]].view(-1, 1)


def bradley_terry_loss(predictions, targets):
    flat_targets = make_square_and_get_triu(targets)
    flat_targets = flat_targets > 0
    flat_targets = flat_targets.float()
    flat_predictions = make_square_and_get_triu(predictions)
    return F.binary_cross_entropy_with_logits(flat_predictions, flat_targets)