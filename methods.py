from enum import IntEnum
import torch


class Method(IntEnum):
    SGD = 0
    ORDERED = 1
    KLDRO = 2
    ZSCORE = 3
    FOCAL = 5
    RANKED = 6


METHOD_NAME_MAP = {
    "sgd": Method.SGD,
    "ordered_sgd": Method.ORDERED,
    "kl_dro": Method.KLDRO,
    "z_score": Method.ZSCORE,
    "focal": Method.FOCAL,
    "rank_based": Method.RANKED,
}

METHOD_VALUE_TO_NAME = {v: k for k, v in METHOD_NAME_MAP.items()}


def parse_method(method_str: str) -> Method:
    return METHOD_NAME_MAP[method_str]


def method_name(method: Method) -> str:
    return METHOD_VALUE_TO_NAME[method]


def select_training_loss(cr_loss, method, ssize):
    method = Method(method)
    bs = cr_loss.size(0)

    if method == Method.SGD:
        return torch.mean(cr_loss)

    elif method == Method.ORDERED:
        if ssize >= bs:
            return torch.mean(cr_loss)
        return torch.topk(cr_loss, k=min(ssize, bs))[0].mean()

    elif method == Method.KLDRO:
        tau = 1.0
        # Normalize losses to prevent numerical overflow in softmax
        cr_loss_norm = cr_loss - cr_loss.max().detach()
        weights = torch.softmax(cr_loss_norm / tau, dim=0)
        return torch.sum(weights * cr_loss)

    elif method == Method.ZSCORE:
        mean = cr_loss.mean()
        std = cr_loss.std() + 1e-8
        z = (cr_loss - mean) / std
        weights = torch.relu(z)
        weights = weights / (weights.sum() + 1e-8)
        return torch.sum(weights * cr_loss)

    elif method == Method.FOCAL:
        gamma = 2.0
        # Clip losses to prevent numerical overflow in power operation
        cr_loss_clipped = torch.clamp(cr_loss, max=10.0)
        weights = cr_loss_clipped ** gamma
        weights = weights / (weights.sum() + 1e-8)
        return torch.sum(weights * cr_loss)

    elif method == Method.RANKED:
        alpha = 2.0
        ranks = torch.argsort(torch.argsort(cr_loss)) + 1
        weights = ranks.float() ** alpha
        weights = weights / weights.sum()
        return torch.sum(weights * cr_loss)

    else:
        raise ValueError(f"Unknown method: {method}")