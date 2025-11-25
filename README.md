# Class Imbalance Loss Comparison
## Objective
Implement and analyze Focal Loss for handling class imbalance. Compare its convergence and final F1-score performance against a standard weighted cross-entropy.

## Roadmap
1. Research about weighted cross-entropy, focal loss, behaviour and $\alpha$-balanced versions
2. Implement the vocal loss
3. Test on artificial dataset (make moons)
4. Evaluate artificial dataset results
5. Create real world example (credit card fraud)
6. 

## Definitions
### Focal Loss

Focal Loss is a modified cross-entropy loss that addresses class-imbalance by down-weighting easy, well-classified examples and focusing training on hard ones. It is defined as:
$$
\text{FL}(p_t) = -\alpha_t \,(1 - p_t)^{\gamma}\;\log(p_t)
$$
where $\alpha_t$ balances classes and $\gamma$ controls the focus on hard samples.  
Introduced in [Lin et al., 2017 — *Focal Loss for Dense Object Detection*](https://arxiv.org/pdf/1708.02002).

### Weighted Cross Entropy
$$
\text{WCE}(p_t) = -\alpha_t \log(p_t)
$$

### Cross Entropy
$$
\text{CE}(p_t) = -\log(p_t)
$$