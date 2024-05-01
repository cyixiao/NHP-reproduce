# Neural Hypergraph Link Prediction (NHP) Reproduction

## Overview
This project aims to reproduce the results of the undirected case of the Neural Hypergraph Link Prediction (NHP) model, based on the research paper (https://dl.acm.org/doi/pdf/10.1145/3340531.3411870). My focus is on implementing the NHP model with two scoring functions, 'mean' and 'maxmin', as discussed in the paper, and evaluating their performance across multiple datasets.

## Datasets (Reactions of chemical reaction networks)
- iAF1260b
- iJO1366
- USPTO (organic reaction dataset)

The datasets used in this project are detailed in data_sample.py, which outlines the characteristics and preliminary statistics of each dataset involved in the study.

## Usage
Run main.py to execute the model training and testing across different dataset splits. This script will:
1. Randomly generate 10 train-test splits for each dataset.
2. Train the NHP model using both `mean` and `maxmin` functions for the Hyperlink scoring layer.
3. Evaluate model performance using AUC and Recall@k metrics, where k is half the number of missing links.

## Results
The results of this experiment are summarized below. Each value represents the mean ± standard deviation computed over 10 random splits.
| Scoring Function | Dataset   | AUC       | Recall@k  |
|------------------|-----------|-----------|-----------|
| mean             | iAF1260b  | 0.58 ± 0.01 | 0.29 ± 0.01 |
|                  | iJO1366   | 0.60 ± 0.01 | 0.29 ± 0.01 |
|                  | uspto     | 0.72 ± 0.01 | 0.36 ± 0.01 |
| maxmin           | iAF1260b  | 0.64 ± 0.01 | 0.33 ± 0.01 |
|                  | iJO1366   | 0.65 ± 0.01 | 0.33 ± 0.01 |
|                  | uspto     | 0.90 ± 0.01 | 0.44 ± 0.00 |
