import torch
import torch.nn.functional as F

def _softplus_evidence(x):
    return F.softplus(x)

def edl_uncertainty_from_evidence(evidence):
    ev = _softplus_evidence(evidence)
    alpha = ev + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    K = alpha.shape[1]
    vacuity = K / (S + 1e-6)
    prob = alpha[:,1:2] / (S + 1e-6)
    return prob, vacuity


def edl_prob_uncert_logit_from_evidence(evidence, eps: float = 1e-6):
    """Return (p_fg, vacuity, logit_fg) from raw evidential outputs.

    This is useful when downstream modules are written in the standard
    "logit -> sigmoid" parameterization, while the base model uses an
    evidential head.

    For binary case with alpha=[alpha_bg, alpha_fg]:
      p_fg = alpha_fg / (alpha_bg + alpha_fg)
      logit(p_fg) = log(alpha_fg) - log(alpha_bg)

    Args:
      evidence: [B,2,H,W] raw evidential outputs (can be any real numbers).
      eps: numerical stability for log.
    Returns:
      p_fg:   [B,1,H,W]
      vacuity:[B,1,H,W]
      logit:  [B,1,H,W] equals logit(p_fg) under the evidential mean.
    """
    ev = _softplus_evidence(evidence)
    alpha = ev + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    K = alpha.shape[1]
    vacuity = K / (S + eps)

    a0 = alpha[:, 0:1]
    a1 = alpha[:, 1:2]
    p_fg = a1 / (S + eps)
    logit = torch.log(a1 + eps) - torch.log(a0 + eps)
    return p_fg, vacuity, logit

def edl_binary_loss(evidence, target, coeff_kl=1e-4):
    ev = _softplus_evidence(evidence)
    alpha = ev + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    prob = alpha / (S + 1e-6)
    T = torch.cat([1.0-target, target], dim=1)
    nll = -(T * torch.log(prob + 1e-6)).sum(dim=1, keepdim=True).mean()
    K = alpha.shape[1]
    sum_alpha = alpha.sum(dim=1, keepdim=True)
    lgamma_sum = torch.lgamma(sum_alpha)
    lgamma = torch.lgamma(alpha).sum(dim=1, keepdim=True)
    digamma = torch.digamma(alpha)
    digamma_sum = torch.digamma(sum_alpha)
    kl = (lgamma_sum - lgamma - torch.lgamma(torch.tensor(float(K), device=alpha.device))
          + (alpha - 1) * (digamma - digamma_sum)).sum(dim=1, keepdim=True).mean()
    return nll + coeff_kl*kl
