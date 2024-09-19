import torch.nn.functional as F
import torch

def jaccard_pow_loss(y_pred, y_true, p_value=1.5, smooth=10, from_logits=True):
    if from_logits:
        y_pred = F.logsigmoid(y_pred.float()).exp()
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    intersection = torch.sum(y_true_f * y_pred_f)
    term_true = torch.sum(y_true_f.pow(p_value))
    term_pred = torch.sum(y_pred_f.pow(p_value))
    union = term_true + term_pred - intersection
    return 1 - ((intersection + smooth) / (union + smooth))

def dice_pow_loss(y_pred, y_true, p_value=1.5, smooth=10, from_logits=True):
    if from_logits:
        y_pred = F.logsigmoid(y_pred.float()).exp()
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)

    numerator = torch.sum(2 * (y_true_f * y_pred_f))
    y_true_f = y_true_f.pow(p_value)
    y_pred_f = y_pred_f.pow(p_value)
    denominator = torch.sum(y_true_f) + torch.sum(y_pred_f)
    return 1 - ((numerator + smooth) / (denominator + smooth ))