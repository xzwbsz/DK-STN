# DK-STN
A Domain Knowledge Embedded Spatio-Temporal Network Model for MJO Forecast

Understanding and predicting the Madden-Julian Oscillation (MJO) is fundamental for forecasting precipitation and preventing disasters. To date, long-term and accurate MJO prediction has been a pending problem for researchers.

This project shows the code of our MJO prediction with domain knowledge embedded spatio-temporal network model.

## Requirements

python == 3.8.0
pytorch == 1.10.1

**Installation**
pip install -r requirement.txt

## Cal RMM

The Code folder contains codes for processing climate data, calculating RMM indices, and sliding windows to build the dataset. (Climate dataset not uploaded because it is too large.)

## Model

The train.py carried out the training of the model (DK-STN), while the testing of the model was carried out in the Test folder, which resulted in 27-28 days, and the best model parameters were stored as the .pth file.

## Update(2023.05.05)

The previous test set had too few cases, so we expanded the test set (i.e., 2019.09.01 to 2023.04.18) and reran the test while ensuring that the predictions were not affected, and the results remained at 25 to 26 days.
