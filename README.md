# LagAttention

LagAttention: Modeling Temporal Asynchrony in Multivariate Time Series Forecasting


## Overview

**LagAttention** is the first time series forecasting framework that models asynchronous interactions among variates. It introduces two key mechanisms to address both micro-level and macro-level asynchrony:
1. **Grouped Synchronization (GS)**: This feature dynamically aligns correlated variates based on their temporal profiles, effectively compensating for micro-level asynchrony.
2. **Temporal-Channel Cross Attention (TCCA)**: A novel attention mechanism that captures macro-level asynchrony across variates over multiple periods, enhancing the forecasting accuracy.
LagAttention provides a robust solution for time series forecasting by integrating these two approaches, ensuring accurate predictions even in the presence of asynchronous variates.
 

## Get Started

1. Install requirements. ```pip install -r requirements.txt```

2. Download data. You can download the all datasets from [datasets](https:.....). **All the datasets are well pre-processed** and can be used easily.

3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
# Long-term Forecasting
bash ./scripts/long_term_forecast/ETT_script/LagAttention_ETTh1.sh
bash ./scripts/long_term_forecast/ETT_script/LagAttention_ETTh2.sh
bash ./scripts/long_term_forecast/ETT_script/LagAttention_ETTm1.sh
bash ./scripts/long_term_forecast/ETT_script/LagAttention_ETTm2.sh
bash ./scripts/long_term_forecast/ECL_script/LagAttention.sh
bash ./scripts/long_term_forecast/Traffic_script/LagAttention.sh
bash ./scripts/long_term_forecast/Solar_script/LagAttention.sh
bash ./scripts/long_term_forecast/Weather_script/LagAttention.sh
bash ./scripts/long_term_forecast/Exchange_script/LagAttention.sh

# Short-term Forecasting
bash ./scripts/short_term_forecast/Illness_script/LagAttention.sh
bash ./scripts/short_term_forecast/PEMS/LagAttention.sh
```


## Key Modules & Implementation

The framework of LagAttention is structured into several key modules, each of which serves a specific purpose. Here are the primary components along with their locations within the codebase:

### 1. Data Provider
**File:** `./data_provider`
This module is responsible for loading and preprocessing the data used in both long-term and short-term forecasting tasks.

### 2. Experiments
**File:** `./exp`
This module contains two main functionalities for experimenting with forecasting:
- `./exp/exp_long_term_forecasting.py`: For long term forecasting.
- `./exp/exp_short_term_forecasting.py`: For short term forecasting.

### 3. Layers
**File:** `./layers`
This module contains layers for different architectures are organized into their respective classes:
- `./layers/AutoCorrelation.py`: For temporal correlation analysis.
- `./layers/Autoformer_EncDec.py`: For autoformer-based models.
- `./layers/Embed.py`: For input representation.
- `./layers/SelfAttention_Family.py`: For attention mechanisms, it implements **Temporal-Channel Cross Attention (TCCA)** of LagAttention.
- `./layers/StandardNorm.py`: For data normalization.
- `./layers/Transformer_EncDec.py`: For transformer-based architectures.

### 4. Models
**File:** `./models.`
- `./exp/LagAttention.py`: Contains various model implementations that leverage the layers defined in `./layers`, which implements **Grouped Synchronization (GS)** of LagAttention.

### 5. Utilities
**File:** `utils.py`
This module provides utility functions that help with other tasks.


## Summary of Structure:

1. **Overview**: A brief description of LagAttention and its purpose.
2. **Get Started**: Detailed steps for installation, data downloading, and model training.
3. **Key Modules & Implementation**: Breakdown of important modules, their corresponding files and descriptions.