# Class Imbalance Loss Comparison
## Objective
Implement and analyze Focal Loss for handling class imbalance. Compare its convergence and final F1-score performance against a standard weighted cross-entropy.
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
> Comment: <br>
> Questions: <br>
> Do we really need the $\alpha$ there? Why? <br>
> Isn't the Focal loss a weighted cross entropy? <br>
> Why does it do what focal loss does? <br>
> How and why is that better than WCE? Why does WCE and CE still exist then? pros/cons <br>
> 
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

# Presentation
1. Task
2. Agenda
1. Motivation
   2. Imbalanced Datasets (Balancing methods for datasets)
   4. Leaving the dataset unbalanced (examples)
   5. Dealing with Imbalanced datasets during training
2. Introduction & Related Work
   3. Loss Types
      3. Normal Loss
      4. Cross Entropy (Log Loss)
      5. Weighted Cross Entropy 
         6. (https://link.springer.com/article/10.1007/s11063-018-09977-1 can't access)
         7. https://arxiv.org/pdf/2006.01413
         8. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10502684
      6. Focal Loss 
         7. (Cite Paper https://arxiv.org/pdf/1708.02002)
         7. https://proceedings.neurips.cc/paper_files/paper/2020/file/aeb7b30ef1d024a76f21a1d40e30c302-Paper.pdf
   7. Compare loss functions mathematically and visually
   8. Explain those loss functions
3. Experiment
   4. Dataset
   5. Setup
   6. Model
   7. Method (MLFlow) & Training method (How we ran the trainings)
   8. Results (visually: make it pretty with graphs)
4. Conclusion
   5. What do the graphs tell us?
   6. Experiments findings
   7. What do those findings mean?
   8. How are they related to the real world examples from the beginning?
   9. Which loss function is better now? When?
   10. What is our message to the world?
12. Sources