# Setup Guide for Table Augmentation Model

## 1. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Note: PyTorch Geometric installation may require specific commands based on your CUDA version
# Visit: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# For CUDA 11.8 (example):
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# For CPU only:
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

## 2. Verify Aim Installation

```bash
# Check Aim version
aim version

# Initialize Aim repository (optional, will be auto-created)
aim init
```

## 3. Start Aim UI (Optional, for monitoring)

```bash
# Start Aim dashboard
aim up

# Or specify custom port
aim up --port 43800

# Access in browser
# http://localhost:43800
```

## 4. Configuration Files

All configuration is in `configs/` directory:

- **data_config.yaml**: Data paths, preprocessing, Layer 2/3 settings
- **model_config.yaml**: Model architecture, Layer 1/2/3 hyperparameters  
- **training_config.yaml**: Training settings, optimizer, Aim logging

### Key Configuration Options

#### Enable/Disable Aim Tracking

```yaml
# In training_config.yaml
logging:
  aim:
    enabled: true  # Set to false to disable
```

#### Adjust Data Sampling

```yaml
# In data_config.yaml
preprocessing:
  max_rows_per_table: 5000  # Increase for more data
```

#### Change Model Size

```yaml
# In model_config.yaml
layer1:
  embedding_dim: 128  # Increase to 256 for larger model
```

## 5. Directory Structure

After setup, your directory should look like:

```
opendata/
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── training_config.yaml
├── src/
│   ├── data/
│   ├── models/
│   ├── training/
│   └── utils/
├── scripts/
├── processed_data/
│   └── cache/
├── aim_logs/  (created automatically)
├── experiments/  (created during training)
└── requirements.txt
```

## 6. Next Steps

See implementation plan for stage-by-stage development:
- Stage 1: Data preparation + Layer 1
- Stage 2: Layer 2 (Joinability)
- Stage 3: Layer 3 (Column augmentation)
- Stage 4: End-to-end integration
- Stage 5: Inference & deployment

## Troubleshooting

### Aim not tracking experiments

```bash
# Check if Aim repo exists
ls aim_logs/

# Reinitialize if needed
rm -rf aim_logs/
aim init
```

### CUDA/GPU issues

```bash
# Check PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If false, you may need to reinstall PyTorch with CUDA support
```

### Memory issues

```yaml
# Reduce batch size in training_config.yaml
training:
  batch_size: 8  # Reduce from 16

# Reduce data size in data_config.yaml
preprocessing:
  max_rows_per_table: 2000  # Reduce from 5000
```

