# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class DBLWithFGW(nn.Module):
    def __init__(
        self,
        # ---- DBL  ----
        m0: float = 0.0,
        lam_dbl: float = 1.0,
        m_max: float = 0.5,
        # ---- FGW  ----
        lambda_fg: float = 1.0,
        beta: float = 0.999,
        lambda_F: float = 1.0,
        lambda_B: float = 1.0,
        tau: float = 0.5,
        class_counts: Optional[torch.Tensor] = None,

        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        reduction: str = "mean",  # 'mean' | 'sum' | 'none'
    ):
        super().__init__()

        # DBL
        self.m0 = float(m0)
        self.lam_dbl = float(lam_dbl)
        self.m_max = float(m_max)

        # FGW
        assert tau > 0.0
        self.lambda_fg = float(lambda_fg)
        self.beta = float(beta)
        self.lambda_F = float(lambda_F)
        self.lambda_B = float(lambda_B)
        self.tau = float(tau)


        self.ignore_index = int(ignore_index)
        self.label_smoothing = float(label_smoothing)
        assert reduction in ("mean", "sum", "none")
        self.reduction = reduction


        if class_counts is not None:
            self.set_class_counts(class_counts)
        else:
            self.register_buffer("class_counts", None)
            self.register_buffer("cb_weight_per_class", None)

    # ---------------- util ----------------
    @torch.no_grad()
    def set_class_counts(self, counts: torch.Tensor) -> None:
        counts = counts.to(torch.float)
        self.register_buffer("class_counts", counts)
        # w_cb(c) = (1 - β) / (1 - β^{n_c})
        cb = (1.0 - self.beta) / (1.0 - torch.clamp(self.beta ** counts, max=0.9999999))
        self.register_buffer("cb_weight_per_class", cb)

    @staticmethod
    def _maybe_scalar(x):
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return float(x.detach().cpu())
        return x

    @staticmethod
    def _safe_t_index(target: torch.Tensor, ignore_index: int) -> torch.Tensor:
        t_idx = target.clone()
        if ignore_index is not None:
            t_idx = torch.where(t_idx == ignore_index, torch.zeros_like(t_idx), t_idx)
        return t_idx

    def _hardest_competitor(self, logits: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask.scatter_(1, t_idx.view(-1, 1), True)
        masked = logits.masked_fill(mask, float("-inf"))
        s_star, _ = masked.max(dim=1, keepdim=True)
        return s_star  # [B,1]

    def _apply_dbl(
        self, logits: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        t_idx = self._safe_t_index(target, self.ignore_index)

        s_y = logits.gather(1, t_idx.view(-1, 1))              # [B,1]
        s_star = self._hardest_competitor(logits, t_idx)       # [B,1]
        margin = self.m0 + self.lam_dbl * torch.sigmoid(s_star - s_y)
        margin = torch.clamp(margin, 0.0, self.m_max)          # [B,1]
        s_prime = logits.clone()
        s_prime.scatter_(1, t_idx.view(-1, 1), (s_y - margin))

        ce_vec = F.cross_entropy(
            s_prime,
            target,
            reduction="none",
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        ce_vec = torch.where(valid_mask, ce_vec, torch.zeros_like(ce_vec))
        return s_prime, ce_vec

    # -------------- forward ----------------
    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        logits: [B, C]
        target: [B]
        返回 (L_total, stats)
        """
        if logits.ndim != 2:
            raise ValueError(f"logits shape 应为 [B,C]，但得到 {logits.shape}")

        B, C = logits.shape
        device = logits.device

        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            zero = logits.sum() * 0.0
            return zero, {"L_total": 0.0, "L_DBL": 0.0, "L_FGW": 0.0}

        s_prime, ce_vec = self._apply_dbl(logits, target, valid_mask)

        if self.reduction == "mean":
            L_DBL = ce_vec[valid_mask].mean()
        elif self.reduction == "sum":
            L_DBL = ce_vec[valid_mask].sum()
        else:
            L_DBL = ce_vec  # [B]

        t_idx = self._safe_t_index(target, self.ignore_index)
        if self.cb_weight_per_class is not None:
            cb_per_class = self.cb_weight_per_class.to(device)
            w_cb = cb_per_class.gather(0, t_idx)  # [B]
        else:
            counts_batch = torch.bincount(target[valid_mask], minlength=C).to(torch.float)
            counts_batch = torch.clamp(counts_batch, min=1.0)
            w_cb_pc = (1.0 - self.beta) / (1.0 - torch.clamp(self.beta ** counts_batch, max=0.9999999))
            w_cb = w_cb_pc.gather(0, t_idx)  # [B]

        prob = F.softmax(s_prime, dim=1)
        p_y = prob.gather(1, t_idx.view(-1, 1)).squeeze(1)  # [B]
        w_F = 1.0 - p_y

        s_y_prime = s_prime.gather(1, t_idx.view(-1, 1))        # [B,1]
        s_star_prime = self._hardest_competitor(s_prime, t_idx) # [B,1]
        delta_prime = (s_y_prime - s_star_prime).squeeze(1)     # [B]
        w_B = torch.sigmoid((self.tau - delta_prime) / self.tau)

        w_tilde = w_cb * (1.0 + self.lambda_F * w_F + self.lambda_B * w_B)  # [B]
        w_tilde = torch.where(valid_mask, w_tilde, torch.zeros_like(w_tilde))
        denom = w_tilde[valid_mask].mean().clamp(min=1e-12)  # = (1/B)sum w̃(b)
        w = w_tilde / denom  # [B]

        L_FGW_vec = w * ce_vec  # [B]
        if self.reduction == "mean":
            L_FGW = L_FGW_vec[valid_mask].mean()
        elif self.reduction == "sum":
            L_FGW = L_FGW_vec[valid_mask].sum()
        else:
            L_FGW = L_FGW_vec

        if self.reduction == "none":
            L_total = L_DBL + self.lambda_fg * L_FGW
        else:
            L_total = L_DBL + self.lambda_fg * L_FGW

        stats: Dict[str, Any] = {
            "L_total": self._maybe_scalar(L_total),
            "L_DBL": self._maybe_scalar(L_DBL),
            "L_FGW": self._maybe_scalar(L_FGW),
            "w_cb_mean": float(w_cb[valid_mask].mean().detach().cpu()),
            "w_F_mean": float(w_F[valid_mask].mean().detach().cpu()),
            "w_B_mean": float(w_B[valid_mask].mean().detach().cpu()),
        }
        return L_total, stats

