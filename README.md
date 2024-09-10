# 📦 Diabetes Prediction System
This project is a machine learning system that predicts whether a person is diabetic based on certain medical inputs. The model is built using a Support Vector Machine (SVM) with a linear kernel, trained on a diabetes dataset. The project also demonstrates how to save and load a trained model using pickle.

## 📊 Dataset
The dataset used is the Pima Indians Diabetes Database, which contains medical diagnostic measurements. The key target variable is the Outcome, which indicates whether a patient is diabetic (1) or non-diabetic (0).

## 💻 Features
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- Blood Pressure: Diastolic blood pressure (mm Hg)
- Skin Thickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- Diabetes Pedigree Function: A function that scores likelihood of diabetes based on family history
- Age: Age in years
  
## 🛠️ Dependencies
The project uses the following libraries:
- numpy
- pandas
- scikit-learn
You can install the dependencies with:
```
pip install numpy pandas scikit-learn
```

## 🚀 Model Training Process
### 1. Data Collection and Preprocessing
Load the diabetes dataset into a pandas DataFrame.
Perform exploratory data analysis (EDA), such as viewing dataset shape, descriptive statistics, and correlations.
Separate the dataset into features (x) and labels (y).
Standardize the feature data using StandardScaler.
### 2. Model Training
Split the dataset into training and test sets (80% training, 20% testing) using stratified sampling.
Train a Support Vector Machine (SVM) with a linear kernel on the training data.
### 3. Model Evaluation
Evaluate the model's accuracy on both the training and test sets using accuracy_score.
### 4. Making Predictions
Build a prediction system to classify a new instance as diabetic or non-diabetic based on user input.
### 5. Saving and Loading the Model
Save the trained SVM model to a .sav file using pickle.
Load the saved model to make future predictions without retraining.

## 📂 Files
diabetes_model.sav: The saved trained model file.
diabetes.ipynb: Script for training and saving the model.
diabetes.csv: The dataset.

## 🔗 Further Reading
[Pima Indians Diabetes Database
](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

[Support Vector Machines
](https://scikit-learn.org/stable/modules/svm.html)

[Pickle Documentation](https://docs.python.org/3/library/pickle.html)
