# Molecular Potency Prediction: A GCN and Classical ML Approach for Drug Analysis

## Project Overview

This project focuses on the crucial task of **molecular potency prediction** within the field of **drug analysis**. The primary objective is to accurately predict the `pIC50` values of chemical compounds. `pIC50` is a widely used measure in pharmacology that quantifies a compound's inhibitory potency against a specific biological target. A higher pIC50 value indicates a more potent compound. The project explores the capabilities of Graph Convolutional Networks (GCNs) for this task, leveraging their ability to learn directly from molecular graph structures, and compares their performance against established machine learning baselines.

## Key Technologies and Libraries

This project utilizes a robust set of Python libraries for data handling, molecular featurization, model building, and visualization:

*   **pandas & numpy:** For efficient data manipulation and numerical operations.
*   **RDKit:** A cheminformatics toolkit essential for handling molecular structures (SMILES strings), generating molecular graphs, and computing molecular fingerprints.
*   **torch & torch_geometric:** The core deep learning frameworks for building and training Graph Convolutional Networks (GCNs).
*   **scikit-learn:** Provides tools for dataset splitting, model selection, and performance metrics calculation.
*   **xgboost & scikit-learn:** For implementing and evaluating classical machine learning baseline models (specifically XGBoost).
*   **matplotlib & seaborn:** For generating insightful visualizations of model predictions, training progress, and comparative analysis.

## Models Implemented

The project evaluates and compares the performance of three distinct modeling approaches:

1.  **Graph Convolutional Network (GCN):** A neural network designed to operate directly on graph-structured data. This model processes molecules as graphs, where atoms are nodes and bonds are edges, allowing it to learn complex structural patterns relevant to potency prediction. Two GCN variants are implemented: a standard GCN and an Improved GCN with attention pooling.

2.  **Random Forest Regressor:** A powerful ensemble learning method that builds multiple decision trees during training and outputs the mean prediction of the individual trees. This model is applied to molecular fingerprints (ECFP4) to predict pIC50.

3.  **XGBoost Regressor:** An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable. Similar to Random Forest, it operates on molecular fingerprints (ECFP4) as a strong classical machine learning baseline.

## Project Pipeline and Methodology

The entire prediction process is encapsulated within a `run_complete_pipeline` function, which orchestrates the following steps:

1.  **Data Loading & Preprocessing:**
    *   Reads molecular data (SMILES and pIC50) from a CSV file.
    *   Converts SMILES strings into graph representations for GCN models, extracting atom features and bond connectivity.
    *   **Crucially, it filters out any molecules with invalid (NaN or Inf) pIC50 target values** to ensure data quality and prevent model instability.
    *   Splits the dataset into training, validation, and testing sets.

2.  **Model Training:**
    *   **GCN Model:** Initializes and trains the selected GCN architecture (GCN, Improved GCN, or GAT) using the molecular graph data. Training incorporates Adam optimizer, Mean Squared Error (MSE) loss, and `ReduceLROnPlateau` scheduler for adaptive learning rate, along with early stopping to prevent overfitting.
    *   **Baseline Models (Random Forest & XGBoost):** Generates ECFP4 molecular fingerprints for the valid molecules. These fingerprints are then used to train and evaluate the Random Forest and XGBoost regressors.

3.  **Evaluation:**
    *   After training, each model generates predictions on the independent test set.
    *   Standard regression metrics are calculated: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score.

4.  **Visualization:**
    *   Generates `Predicted vs. Actual` plots to visually assess the correlation between true and predicted pIC50 values for each model.
    *   Displays `Training and Validation Loss` curves for the GCN to monitor training progress.
    *   Creates `Residual Plots` to analyze the distribution of prediction errors.
    *   Provides `Model Comparison` bar charts for R², RMSE, and MAE to facilitate easy comparison across all implemented models.

## Key Findings (Example - based on previous output)

In the latest execution, the comparative analysis revealed significant differences in performance:

*   **XGBoost** achieved the highest R² score (~0.9567), indicating excellent predictive power on the test set.
*   **Random Forest** followed closely with a strong R² score (~0.9520).
*   The **GCN** model showed a low R² score (~0.0004), suggesting it did not learn effective representations for pIC50 prediction in this configuration. This might warrant further hyperparameter tuning, architectural adjustments, or larger/different datasets.

This project provides a clear framework for molecular potency prediction and highlights the current efficacy of fingerprint-based classical machine learning models compared to the initial GCN implementation for this specific dataset and setup.
