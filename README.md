# Class Imbalance Loss Comparison
## Objective
Implement and analyze Focal Loss for handling class imbalance. Compare its convergence and final F1-score performance against a standard weighted cross-entropy.

## Definitions
### Focal Loss
Focal Loss is a modified cross-entropy loss that addresses class-imbalance by down-weighting easy, well-classified examples and focusing training on hard ones. It is defined as:

$$
\mathrm{FL}(p_t) = -\alpha_t \(1 - p_t)^{\gamma}\ \log(p_t)
$$

where $\alpha_t$ balances classes and $\gamma$ controls the focus on hard samples. <br>
Introduced in [Lin et al., 2017 — *Focal Loss for Dense Object Detection*](https://arxiv.org/pdf/1708.02002).

### Weighted Cross Entropy
We add a weightning factor $\omega$ to re-balance the minority class during training for imbalanced datasets. <br>

$$
\mathrm{WCE}(y, \hat{y}) = -\left(\omega \cdot y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right)
$$

Details: [Learning from Imbalanced Data Sets with Weighted Cross-Entropy Function](https://doi.org/10.1007/s11063-018-09977-1)

### Cross Entropy
The normal binary cross entropy used for classification tasks. Here, both classes are weighted the same, indendent wether there is an imbalance in the dataset or not.

$$
\mathrm{BCE}(y, \hat{y}) = -\left( y \cdot \log(\hat{y}) + (1 - y) \cdot \log(1 - \hat{y}) \right)
$$

## Repository layout
- `Moons/`: Synthetic moons experiments with configurable imbalance and noise levels. Includes the focal loss implementation (`focal_loss.py`), dataset builder (`dataset.py`), model definition (`model.py`), and training script (`train.py`).
- `Creditcard/`: Experiments on the credit card fraud dataset with configurable subsampling to control imbalance. Uses the shared `focal_loss.py` and `model.py` modules beside `train.py`.
- `Visulaisation/`: Notebooks and a small script for inspecting data and focal loss behaviour.
- `pyproject.toml`: Python dependencies and basic project metadata.

## Running experiments
### Synthetic moons dataset
1. Adjust hyperparameters and loss settings in `Moons/experiment_config.yaml` if needed.
2. Launch training:
   ```bash
   python Moons/train.py
   ```

### Credit card fraud dataset
1. Download `creditcard.csv` [Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) into the `Creditcard/` directory.
2. Update `Creditcard/experiment_config.yaml` for batch size, epochs, loss choices, and subsampling ratios.
3. Adjust mlflow `PORT` settings in `train.py`
4. Start training:
   ```bash
   python Creditcard/train.py
   ```
## Outputs and visualization
- The `Visulaisation/` notebooks (`data_viewer_creditcard.ipynb`, `data_viewer_moons.ipynb`, `dataset_viewer_moons.ipynb`) help inspect dataset balance and model outputs.

