# Class Imbalance Loss Comparison
## Objective
Implement and analyze Focal Loss for handling class imbalance. Compare its convergence and final F1-score performance against a standard weighted cross-entropy.
## Definitions
### Focal Loss

Focal Loss is a modified cross-entropy loss that addresses class-imbalance by down-weighting easy, well-classified examples and focusing training on hard ones. It is defined as:

$$
\text{FL}(p_t) = -\alpha_t \(1 - p_t)^{\gamma}\ \log(p_t)
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
## Repository layout
- `Moons/`: Synthetic two-moons experiments with configurable imbalance and noise levels. Includes the focal loss implementation (`focal_loss.py`), dataset builder (`dataset.py`), model definition (`model.py`), and training script (`train.py`).
- `Creditcard/`: Experiments on the credit card fraud dataset with configurable subsampling to control imbalance. Uses the shared `focal_loss.py` and `model.py` modules beside `train.py`.
- `Visulaisation/`: Notebooks and a small script for inspecting data and focal loss behaviour.
- `pyproject.toml`: Python dependencies and basic project metadata.

## Running experiments
### Synthetic two-moons dataset
1. Adjust hyperparameters and loss settings in `Moons/experiment_config.yaml` if needed.
2. Launch training:
   ```bash
   python Moons/train.py
   ```

### Credit card fraud dataset
1. Download `creditcard.csv` into the `Creditcard/` directory.
2. Update `Creditcard/experiment_config.yaml` for batch size, epochs, loss choices, and subsampling ratios.
3. Start training:
   ```bash
   python Creditcard/train.py
   ```
## Outputs and visualization
- The `Visulaisation/` notebooks (`data_viewer_creditcard.ipynb`, `data_viewer_moons.ipynb`, `dataset_viewer_moons.ipynb`) help inspect dataset balance and model outputs.

