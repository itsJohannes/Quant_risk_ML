# Core Libraries
import pandas as pd
import numpy as np
import zipfile

# Model Evaluation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, fbeta_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import precision_recall_curve

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Resampling Techniques
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# PCA
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Scaling and other pre-processing
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import pointbiserialr


###############################################
# Data preprocessing
###############################################


# split -> resampling -> scaling

# split -> scaling -> try different resampling methods on logistic regression to see which is best ->  resample un-scaled split -> scale -> model logistic regression and knn -> model the rest on resampled og un-scaled split.
# scaling separately for each one.

# Read data from the zip file
with zipfile.ZipFile('C:/Users/b407939/Documents/Skole/datasets/credit_risk_for_exam.zip', 'r') as z:
    train = z.open('train.csv')
    test = z.open('test.csv')  # Test set for competition. Irrelevant.
    data = pd.read_csv(train, index_col=0)
    test_data = pd.read_csv(test, index_col=0)

# Clean column names
data.columns = data.columns.str.strip()
test_data.columns = test_data.columns.str.strip()

# Export the CSV to Excel
#file_path = "C:/Users/b407939/Documents/Skole/Output/exam_credit_risk.xlsx"
#data.to_excel(file_path, index=None)

# Create dummy variables for non-ordinal categorical features (fuck dummy variables.)
categorical_cols = ['loan_intent']
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)


#__________________________________________________________________________________
# Create label encoding for ordinal categorical features

# Label encode 'person_home_ownership' (Ordinal: rent=0, mortgage=1, own=2)
home_ownership_mapping = {'RENT': 0, 'MORTGAGE': 1, 'OWN': 2, 'OTHER': 3}
data['person_home_ownership'] = data['person_home_ownership'].map(home_ownership_mapping)

# Label encode 'loan_grade' (Ordinal: G=0, F=1, E=2, D=3, C=4, B=5, A=6)
grade_mapping = {'G': 0, 'F': 1, 'E': 2, 'D': 3, 'C': 4, 'B': 5, 'A': 6}
data['loan_grade'] = data['loan_grade'].map(grade_mapping)

# Label encode 'cb_person_default_on_file' (Binary: Y = 1, N = 0)
default_mapping = {'Y': 1, 'N': 0}
data['cb_person_default_on_file'] = data['cb_person_default_on_file'].map(default_mapping)
#___________________________________________________________________________________

# Separate features (X) and target (y)
X = data.drop(['loan_status', 'cb_person_cred_hist_length', 'loan_grade'], axis=1)
y = data['loan_status']

# Split data into training, validation, and test sets
X_train, X_temp, Y_train, Y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)


###############################################
# CORRELATION PLOT TO SEE MULTICOLLINEARITY
###############################################

import seaborn as sns
import matplotlib.pyplot as plt

# Identify categorical columns
categorical_cols = ['loan_intent_EDUCATION',
       'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
       'loan_intent_PERSONAL', 'loan_intent_VENTURE']

# Identify numerical columns by excluding categorical columns
numerical_cols = [col for col in X_train.columns if col not in categorical_cols]

# Compute the correlation matrix for numerical variables
corr_matrix_num = X_train[numerical_cols].corr()

# Create the heatmap for numerical variables
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_num, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Numerical Features')
#plt.show()

# Compute the correlation matrix for categorical variables (dummy variables)
corr_matrix_cat = X_train[categorical_cols].corr()

# Create the heatmap for categorical variables
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix_cat, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix for Categorical Features (Dummy Variables)')
#plt.show()


#____________________________________________
# Compute the point-biserial correlation between categorical and numerical variables 


# Assuming X_train is your feature matrix and numerical_cols, categorical_cols are defined
# Initialize an empty list to store correlation results
correlation_results = []

# Step 1: Calculate the point-biserial correlation for each pair
for cat_col in categorical_cols:
    for num_col in numerical_cols:
        corr, p_value = pointbiserialr(X_train[cat_col], X_train[num_col])
        # Store the results in the list
        correlation_results.append({
            'Categorical Variable': cat_col,
            'Numerical Variable': num_col,
            'Point-Biserial Correlation': corr,
            'p-value': p_value
        })

# Step 2: Convert the list of results to a DataFrame
correlation_df = pd.DataFrame(correlation_results)

# Step 3: Pivot the DataFrame to create a matrix for easier plotting                                             
pivot_df = correlation_df.pivot(index='Categorical Variable', columns='Numerical Variable', values='Point-Biserial Correlation')

# Step 4: Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.title('Point-Biserial Correlation Matrix')
plt.tight_layout()
#plt.show()

###### COMMENTS ON RESULTS ######

# Correlation matrix of Numerical features shows that 'person_age' and 'person_cred_hist_length' are heavily correlated.
# Also shows that 'loan_int_rate' and 'loan_grade' are heavily correlated

# Categorical features correlation matrix does not show any correlation

# Point Biserial correlation matrix does not show any correlation.


# I need to exclude the variables "person_cred_hist_length" and 'loan_grade'.
# 'loan_percent_income' is debateable.


###############################################
# SCALING THE DATA FOR THE KNN MODEL
###############################################

""" # Manually define the columns to be scaled
numerical_cols = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate'] 
# I am not scaling the loan as percentage of income variable since this is already between 0 and 1.

# Create a copy of the original data to avoid modifying it directly
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Initialize the scaler
scaler = StandardScaler()

# Apply the scaling to the numerical columns only
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Convert scaled data back to a DataFrame
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

# Print the first few rows of the DataFrame
print(X_train_scaled_df.head())
 """


###############################################
# FINDING THE BEST RESAMPLING METHOD
###############################################
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
import pandas as pd

# Define the resampling methods
resamplers = {
    'No Resampling': None,
    'Random Oversampling': RandomOverSampler(random_state=42),
    'SMOTE': SMOTE(random_state=42),
    'Random Undersampling': RandomUnderSampler(random_state=42)
    #'Condensed Nearest Neighbour': CondensedNearestNeighbour(random_state=42)
}

# Initialize a dictionary to store results
results = []

# Iterate over each resampling method
for method, resampler in resamplers.items():
    if resampler is None:
        # No resampling
        X_resampled, y_resampled = X_train, Y_train
    else:
        # Apply resampling
        X_resampled, y_resampled = resampler.fit_resample(X_train, Y_train)
    
    # Train the logistic regression model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_resampled, y_resampled)
    
    # Make predictions
    y_pred = model.predict(X_val)
    
    # Calculate F2-scores for both classes
    f2_minority = fbeta_score(Y_val, y_pred, beta=2, pos_label=1)
    f2_majority = fbeta_score(Y_val, y_pred, beta=2, pos_label=0)
    
    # Store the results
    results.append({
        'Resampling Method': method,
        'F2 Score (Minority Class)': f2_minority,
        'F2 Score (Majority Class)': f2_majority,
        'Average F2 Score': (f2_minority + f2_majority) / 2
    })

# Convert results to a DataFrame for comparison
results_df = pd.DataFrame(results)

# Display results
print(results_df)


# Save the results to a Excel file
file_path = 'C:/Users/b407939/Documents/Skole/Output/resampling_comparison.xlsx'
results_df.to_excel(file_path, index=False)

# With random oversampling we get the highest average F2 score.
# Lets do random oversampling
resampler = RandomUnderSampler(random_state=42)
X_train_res, Y_train_res = resampler.fit_resample(X_train, Y_train)



###############################################
# FINDING THE BEST MODEL
###############################################

# Define the models to test
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42), # max iter
    'Decision Tree': DecisionTreeClassifier(random_state=42), # max depth
    'Random Forest': RandomForestClassifier(n_estimators=500, random_state=42), # max depth
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, random_state=42), 
    'Tuned Gradient Boosting': GradientBoostingClassifier(learning_rate=0.2, max_depth=3, min_samples_leaf=2, 
                                                          min_samples_split=2, n_estimators=500, random_state=42)
    #'Support Vector Machine (SVM)': SVC(kernel="rbf", probability=True, random_state=42, max_iter=2000),
    #'K-Nearest Neighbors': KNeighborsClassifier()
}

# Initialize a list to store results
model_results = []

# Iterate over each model
for model_name, model in models.items():
    # Train the model
    model.fit(X_train_res, Y_train_res)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Calculate other metrics
    report = classification_report(Y_test, y_pred, output_dict=True)
    
    # F1 Scores for minority and majority class
    f2_minority = fbeta_score(Y_test, y_pred, beta=2, pos_label=1)
    f2_majority = fbeta_score(Y_test, y_pred, beta=2, pos_label=0)
    
    # Precision and Recall for both classes
    precision_minority = precision_score(Y_test, y_pred, pos_label=1)
    recall_minority = recall_score(Y_test, y_pred, pos_label=1)
    precision_majority = precision_score(Y_test, y_pred, pos_label=0)
    recall_majority = recall_score(Y_test, y_pred, pos_label=0)
    
    # AUC (Area Under the ROC Curve)
    auc = roc_auc_score(Y_test, y_pred_prob) if y_pred_prob is not None else None
    
    # Accuracy of the model
    accuracy = accuracy_score(Y_test, y_pred)
    
    # Store the results
    model_results.append({
        'Model': model_name,
        'F2 Score (Minority Class)': f2_minority,
        'F2 Score (Majority Class)': f2_majority,
        'Average F2 Score': (f2_minority + f2_majority) / 2,
        'Precision (Minority Class)': precision_minority,
        'Recall (Minority Class)': recall_minority,
        'Precision (Majority Class)': precision_majority,
        'Recall (Majority Class)': recall_majority,
        'ROC-AUC': auc,
        'Accuracy': accuracy
    })

############################################################
############################################################
############################################################
# Fine tune GB model on validation set

""" from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Custom F2 scorer
f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1)

# Define the hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the GradientBoostingClassifier
gb_model = GradientBoostingClassifier(random_state=42)

# Use GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid,
    scoring=f2_scorer,  # Use custom F2 scoring
    cv=3,               # 3-fold cross-validation
    verbose=2,
    n_jobs=-1           # Use all available processors
)

print("Starting hyperparameter tuning...")
# Fit the grid search to the resampled training data
grid_search.fit(X_train_res, Y_train_res)

# Display the best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best F2 Score (Validation):", grid_search.best_score_)

# Use the best estimator for further evaluation
best_gb_model = grid_search.best_estimator_

print("Evaluating the best model on the validation set...")
# Predict on the test set
y_pred = best_gb_model.predict(X_val)
y_pred_prob = best_gb_model.predict_proba(X_val)[:, 1]
 """
#729 total fits at 3 folds meaning 243 different candidates for best model.
#Best Parameters: {'learning_rate': 0.2, 'max_depth': 3, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 500}
#Best F2 Score (Validation): 0.8647829223550817
############################################################
############################################################
############################################################

""" # Initialize best model
best_gb_model = GradientBoostingClassifier(learning_rate=0.2, max_depth=3, min_samples_leaf=2, min_samples_split=2,
                                            n_estimators=500, random_state=42)

# Fit model on resampled training data
best_gb_model.fit(X_train_res, Y_train_res)

# Add results for the best Gradient Boosting model to the table
# Predict on the test set
y_pred = best_gb_model.predict(X_test)
y_pred_prob = best_gb_model.predict_proba(X_test)[:, 1]

# Calculate F2 scores
f2_minority = fbeta_score(Y_test, y_pred, beta=2, pos_label=1)
f2_majority = fbeta_score(Y_test, y_pred, beta=2, pos_label=0)

# Calculate precision and recall for both classes
precision_minority = precision_score(Y_test, y_pred, pos_label=1)
recall_minority = recall_score(Y_test, y_pred, pos_label=1)
precision_majority = precision_score(Y_test, y_pred, pos_label=0)
recall_majority = recall_score(Y_test, y_pred, pos_label=0)

# Calculate AUC (Area Under the ROC Curve)
auc = roc_auc_score(Y_test, y_pred_prob)

# Calculate accuracy
accuracy = accuracy_score(Y_test, y_pred)

# Store the tuned Gradient Boosting results
model_results.append({
    'Model': 'Tuned Gradient Boosting',
    'F2 Score (Minority Class)': f2_minority,
    'F2 Score (Majority Class)': f2_majority,
    'Average F2 Score': (f2_minority + f2_majority) / 2,
    'Precision (Minority Class)': precision_minority,
    'Recall (Minority Class)': recall_minority,
    'Precision (Majority Class)': precision_majority,
    'Recall (Majority Class)': recall_majority,
    'ROC-AUC': auc,
    'Accuracy': accuracy
})

# Convert the results into a DataFrame
results_df = pd.DataFrame(model_results)

# Print the results (optional)
print(results_df)
 """
# funny enough it did worse than non-tuned GB model (or just par). Likely because it slightly overfit to the validation set or resampled data.
# or maybe just sacrificed precision a bit too much.
############################################################
############################################################
############################################################


# Save the results to a Excel file
file_path = 'C:/Users/b407939/Documents/Skole/Output/UNDER_model_comparison.xlsx'
results_df.to_excel(file_path, index=False)

# Print the results
print(results_df)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Initialize the plot
plt.figure(figsize=(10, 8))

# Iterate over each model to plot the ROC curve
for model_name, model in models.items():
    # Train the model on PCA-transformed data
    model.fit(X_train_res, Y_train_res)
    
    # Get predicted probabilities for the positive class
    y_pred_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(Y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot the ROC curve for the current model
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot the diagonal line (no discrimination)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

# Labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

# Show the plot
plt.show()


# False positive = When a loan is predicted to default but it doesn't. Missed bussiness opportunity
# False negative = When a loan is predicted non-default, but it does default. Financial loss

# Specificity and recall are the same thing, but for different classes.
