# Class Imbalance Loss Comparison
## Objective
Implement and analyze Focal Loss for handling class imbalance. Compare its convergence and final F1-score performance against a standard weighted cross-entropy.

## Roadmap
- 

## To-Do List

- [ ] Implement Focal Loss - **Jannis**
- [ ] Implement Weighted Cross-Entropy (Make use of PyTorch's WCE)- **Yannick**
- [ ] Define Dataset - **tbd.**
- [ ] Setup first training pipline - **tbd.**
- [ ] Implement framework for experiment tracking - **tbd.**

## Definitions
### Focal Loss

Focal Loss is a modified cross-entropy loss that addresses class-imbalance by down-weighting easy, well-classified examples and focusing training on hard ones. It is defined as:

$$
\text{FL}(p_t) = -\alpha_t \,(1 - p_t)^{\gamma}\;\log(p_t)
$$

where $\alpha_t$ balances classes and $\gamma$ controls the focus on hard samples.  
Introduced in [Lin et al., 2017 — *Focal Loss for Dense Object Detection*](https://arxiv.org/pdf/1708.02002).