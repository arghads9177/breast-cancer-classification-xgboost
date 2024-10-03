# Breast Cancer Classification

## Project Overview

This project involves building a classification model to predict whether a breast cancer tumor is benign or malignant based on features extracted from a digitized image of a fine needle aspirate (FNA) of a breast mass. The dataset contains real-valued features that describe characteristics of the cell nuclei present in the image. The classification task involves predicting the diagnosis (benign or malignant) using these features.

The dataset used in this project is the **Breast Cancer Wisconsin (Diagnostic) Dataset**, which is publicly available on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). The project involves data preprocessing, feature analysis, and building a machine learning model to accurately classify the tumors.

## Dataset Description

The dataset contains the following attributes:

1. **ID number**: Unique identifier for each sample.
2. **Diagnosis**: Target variable (M = malignant, B = benign).
3. **Features**: There are 30 real-valued features computed for each cell nucleus in the image. These features include:
   - **Radius** (mean of distances from center to points on the perimeter)
   - **Texture** (standard deviation of gray-scale values)
   - **Perimeter**
   - **Area**
   - **Smoothness** (local variation in radius lengths)
   - **Compactness** (perimeter² / area - 1.0)
   - **Concavity** (severity of concave portions of the contour)
   - **Concave points** (number of concave portions of the contour)
   - **Symmetry**
   - **Fractal dimension** ("coastline approximation" - 1)
   
   The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in a total of 30 features. For instance:
   - Field 3 is Mean Radius.
   - Field 13 is Radius Standard Error (SE).
   - Field 23 is Worst Radius.

- **Number of Instances**: 569
- **Number of Attributes**: 32 (ID number, diagnosis, and 30 real-valued features)
- **Missing Values**: None
- **Class Distribution**: 357 benign, 212 malignant

## Attribute Information

1. **ID number**
2. **Diagnosis**: (M = malignant, B = benign)
3. **Ten real-valued features** are computed for each cell nucleus:
   - **Radius** (mean of distances from center to points on the perimeter)
   - **Texture** (standard deviation of gray-scale values)
   - **Perimeter**
   - **Area**
   - **Smoothness** (local variation in radius lengths)
   - **Compactness** (perimeter² / area - 1.0)
   - **Concavity** (severity of concave portions of the contour)
   - **Concave points** (number of concave portions of the contour)
   - **Symmetry**
   - **Fractal dimension** ("coastline approximation" - 1)

## Objective

The objective of this project is to:
1. Understand the dataset and perform exploratory data analysis (EDA).
2. Preprocess the data and handle any missing or inconsistent values.
3. Build a classification model using machine learning techniques to predict the diagnosis of breast cancer tumors.
4. Evaluate the model using classification metrics such as accuracy, precision, recall, and F1-score.
5. Improve model performance using techniques like feature selection, hyperparameter tuning, and cross-validation.

## Steps Involved

1. **Data Exploration**:
   - Perform exploratory data analysis to understand the distribution of the features.
   - Visualize the relationships between features using pair plots and correlation heatmaps.
   
2. **Data Preprocessing**:
   - Handle any missing or incorrect values (if necessary).
   - Normalize/standardize the features to improve model performance.
   - Split the data into training and testing sets.

3. **Model Building**:
   - Train with the classification model:
     - XGBoost Classifier
   
4. **Model Evaluation**:
   - Use classification metrics such as accuracy, precision, recall, and F1-score to evaluate model performance.
   - Use a confusion matrix to visualize the true positives, false positives, true negatives, and false negatives.
   - Perform cross-validation to ensure the model generalizes well on unseen data.

5. **Model Optimization**:
   - Use Hyperparameter tuning to enhance the performance of the model.

## Installation

To run this project on your local machine, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-classification.git

## Project Structure
  -  data/: Contains the Breast Cancer Wisconsin (Diagnostic) Dataset in CSV format.
  - notebooks/: Jupyter notebooks for data exploration, preprocessing, and model training.
  - models/: Trained models for predicting breast cancer diagnosis.
  - README.md: Project documentation.

## License
This project is licensed under the GNU Public License - see the LICENSE file for details.

## Contact
For any questions or suggestions, feel free to contact me at [email2argha@gmail.com].
