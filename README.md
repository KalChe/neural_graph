# NeuralGraph

Deep learning model for EEG seizure detection using spatio-temporal neural networks.

## Features

- CNN-based temporal feature extraction
- Focal loss for class imbalance
- ARSVD model compression
- RL-based personalization agent

## Requirements

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py --epochs 40 --batch-size 32 --lr 5e-4
```

## Evaluation

```bash
python eval.py --threshold 0.3
```

## Model Architecture

- Input: 22-channel EEG signals
- Encoder: Multi-layer 1D CNN with batch normalization
- Output: Binary classification (seizure/non-seizure)

## Results

Results are saved to `outputs/evaluation_results.json` with metrics including sensitivity, specificity, and AUC-ROC.

## License

MIT
