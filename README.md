# Class Imbalance Loss Comparison
## Objective
Analyze if Focal Loss provides better F1-scores on an imbalanced binary classification task compared to Weighted Cross-Entropy (WCE) and a baseline.

## Definitions
### Focal Loss

Focal Loss is a modified cross-entropy loss that addresses class-imbalance by down-weighting easy, well-classified examples and focusing training on hard ones. It is defined as:

$$
\text{FL}(p_t) = -\alpha_t \,(1 - p_t)^{\gamma}\;\log(p_t)
$$

where $\alpha_t$ balances classes and $\gamma$ (typically ≈ 2) controls the focus on hard samples.  
Introduced in [Lin et al., 2017 — *Focal Loss for Dense Object Detection*](https://arxiv.org/pdf/1708.02002).