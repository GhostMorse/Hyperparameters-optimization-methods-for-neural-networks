import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, accuracy_score

from .config import device
from .utils import set_weights_from_vec


@torch.no_grad()
def eval_model(model, loader, task_type, y_mean=None, y_std=None):
    model.eval()
    if task_type == 'classification':
        crit = nn.CrossEntropyLoss(reduction='sum')
        total_loss = 0.0
        all_preds, all_true = [], []
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += crit(out, y).item()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_true.append(y.cpu().numpy())
            n += y.size(0)
        mean_loss = total_loss / n
        acc = accuracy_score(np.concatenate(all_true), np.concatenate(all_preds))
        return {"loss": float(mean_loss), "accuracy": float(acc)}
    else:
        crit = nn.MSELoss(reduction='sum')
        total_loss = 0.0
        all_pred, all_true = [], []
        n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_loss += crit(out, y.view_as(out)).item()
            all_pred.append(out.cpu().numpy().ravel())
            all_true.append(y.cpu().numpy().ravel())
            n += y.numel()
        mean_loss = total_loss / n
        pred = np.concatenate(all_pred)
        true = np.concatenate(all_true)
        if y_mean is not None and y_std is not None:
            pred_orig = pred * y_std + y_mean
            true_orig = true * y_std + y_mean
            r2 = r2_score(true_orig, pred_orig)
        else:
            r2 = r2_score(true, pred)
        return {"loss": float(mean_loss), "r2": float(r2)}


def eval_fitness_on_val(vec, model, val_loader, task_type, y_mean=None, y_std=None):
    """Fitness для DE: loss на полном val-сплите."""
    set_weights_from_vec(model, vec)
    metrics = eval_model(model, val_loader, task_type, y_mean, y_std)
    return metrics["loss"]
