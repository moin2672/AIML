Phase 1: Initial Data Preparation (Before Train-Test Split)
These steps are performed on the entire dataset because they typically involve global data quality issues or transformations that don't depend on statistical properties learned from specific data subsets.

Load Data

Action: Read your raw dataset into a suitable data structure, like a Pandas DataFrame.
Why: You need your data in memory to begin any processing.
Remove Duplicates

Action: Identify and remove exact duplicate rows from your entire dataset.
Why: Duplicate observations can unfairly bias your analysis and model training, leading to overly optimistic performance metrics. Removing them ensures each observation is unique.
Initial Missing Value Handling (Deletion Strategy)

Action: At this stage, you might remove columns or rows with a very high percentage of missing values (e.g., columns that are 80-90% NaNs, or rows that are almost entirely NaNs). This is a structural decision about data quality.
Why: These parts of the data are often unrecoverable or provide minimal information, and removing them early can simplify subsequent steps.
Note: Imputation of missing values (e.g., replacing NaNs with mean/median/mode) should NOT happen here. That comes after the train-test split to prevent leakage.
encode_features (Categorical Encoding - Careful Application)

Action: Convert categorical features (object or category dtype) into numerical representations (e.g., One-Hot Encoding, Ordinal Encoding).
Why: Most machine learning algorithms require numerical input.
Data Leakage Concern & Best Practice: While your encode_features function is designed to take a full DataFrame, for strict data leakage prevention with encoders that learn categories (like OneHotEncoder or OrdinalEncoder when categories='auto'), the fit step of these encoders ideally occurs after the train-test split, on the training data only. However, if your categories are known and fixed beforehand, or if you're comfortable that the handle_unknown='ignore' in OneHotEncoder and unknown_value=-1 in OrdinalEncoder adequately prevent issues, you could run it here.
Recommendation for Safety: I will place test_train_split next, implying that advanced encoding that learns from data should happen after the split. If your encode_features is primarily for simple conversion (e.g., pre-defined ordinal mappings or if OneHotEncoder handles unknown categories robustly), it could conceptually fit here. For this sequence, we will assume encoding of categorical features that learn from the data will happen after the split as part of preprocess_data.
Phase 2: Data Splitting (The Most Crucial Step for Leakage Prevention)
This step fundamentally separates your data into what the model can learn from and what it will be evaluated on.

test_train_split
Action: Divide your pre-processed dataset into X_train, X_test, y_train, and y_test.
Why: This is the cornerstone of preventing data leakage. The X_test and y_test sets must remain untouched by any data transformation or model learning that derives parameters from the data, acting as truly unseen data for final evaluation.
Phase 3: Preprocessing on Training Data (Fit on Train, Transform on Train & Test)
All subsequent preprocessing steps that involve learning parameters (like means, standard deviations, min/max values, outlier thresholds, or feature selection logic) must be performed only on the training data (X_train) and then applied to both X_train and X_test.

preprocess_data (Comprehensive Feature Engineering & Remaining Imputation)

Action: This is a broad category. It includes:
Imputation of Missing Values: Now, you'd apply sophisticated imputation strategies (mean, median, mode, K-NN imputation, etc.).
Crucial: Fit the imputer on X_train (imputer.fit(X_train)) and then transform both X_train and X_test (imputer.transform(X_train), imputer.transform(X_test)).
Encoding of Categorical Features (if not done globally before split): If your encode_features function takes X and y separately, or if it uses encoders that learn categories (OneHotEncoder for categories, OrdinalEncoder), this is where you would fit the encoder on X_train and transform both X_train and X_test.
Feature Engineering: Creating new features from existing ones (e.g., combining columns, polynomial features, datetime features). This should be applied consistently to both train and test sets.
Standardization/Scaling: (If not done via preprocess_data as a single step, see step 7/8).
z_outlier_remove / cap_outliers_remove (Numerical Features Only)

Action: Handle outliers in your numerical features.
For z_outlier_remove: Calculate mean and standard deviation only from X_train to determine Z-scores and removal thresholds. Then apply removal/filtering to both X_train and X_test based on these X_train-derived thresholds.
For cap_outliers_remove: Calculate percentile values (e.g., 5th and 95th percentiles) only from X_train. Then apply these capping values to both X_train and X_test.
Why: Outliers can distort statistical measures and impact model training. These processes are data-dependent, so they must be done after the split.
Standardization (e.g., StandardScaler, MinMaxScaler)

Action: Scale your numerical features.
Why: Many algorithms (linear models, neural networks, SVMs, k-NN) perform better when numerical features are on a similar scale.
Crucial: Fit the scaler only on X_train (scaler.fit(X_train_numerical)). Then, transform both X_train_numerical and X_test_numerical using this fitted scaler.
select_features_boruta (or other Feature Selection method like select_features_shap)

Action: Apply your chosen feature selection method.
Why: Reduces dimensionality, removes irrelevant/redundant features, potentially improves model performance, reduces training time, and enhances interpretability.
Crucial: This step must be performed on X_train (and y_train). The Boruta algorithm or SHAP importance calculation will determine which features are important based only on the training data. Once the set of selected features is identified, you then subset both X_train and X_test to include only those selected features.
Phase 4: Model Development (Using Training Data)
These steps involve building and refining your machine learning model.

Model Creation

Action: Choose a machine learning algorithm (e.g., LGBMClassifier, RandomForestClassifier, LogisticRegression). Instantiate the model with default parameters or initial guesses.
Why: This is the core of your predictive system.
Model Hyperparameter Fine-Tuning

Action: Use techniques like GridSearchCV, RandomizedSearchCV, or more advanced optimization methods (e.g., Optuna, Hyperopt) to find the best set of hyperparameters for your chosen model.
Why: Optimizing hyperparameters improves your model's performance.
Crucial: This tuning must be done using cross-validation on the training data only. You should never use the test set for hyperparameter tuning, as this would lead to leakage.
Model Selection

Action: If you've trained multiple models (e.g., LGBMClassifier, XGBoost, RandomForest) or different versions of the same model with fine-tuning, this is where you choose the best performing model based on cross-validation results from your training set.
Why: To pick the most promising model for final evaluation.
Phase 5: Final Evaluation
Final Model Evaluation on Test Set
Action: Take your best performing, hyperparameter-tuned model and evaluate its performance once on the untouched X_test and y_test data.
Why: This provides an unbiased estimate of how your model will perform on truly unseen data in a real-world scenario. If you evaluate on the test set multiple times or use it for tuning, this estimate becomes unreliable.
Summary of the Most Critical Data Leakage Points:

Train-Test Split: Always the first step after initial data loading and duplicate removal.
Any fit() operation on a transformer/scaler/imputer/feature selector: Must be done only on X_train.
Any transform() operation: Must be done on both X_train and X_test using the same object that was fitted on X_train.
Hyperparameter Tuning & Model Selection: Must be done using cross-validation on the training set only.
Final Evaluation: Only one final evaluation on the untouched test set.
By following this sequence, you build a robust and reliable machine learning pipeline.






