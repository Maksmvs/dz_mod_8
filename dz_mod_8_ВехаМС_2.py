import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC  # Доданий імпорт SVM-класифікатора
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.impute import SimpleImputer

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


train_data_path = r'K:\PowerBi\Go IT Magistr\8_Machine Learning Fundamentals and Applications\dz_mod_8\mod_04_hw_train_data.csv'
valid_data_path = r'K:\PowerBi\Go IT Magistr\8_Machine Learning Fundamentals and Applications\dz_mod_8\mod_04_hw_valid_data.csv'

train_data = pd.read_csv(train_data_path)
valid_data = pd.read_csv(valid_data_path)

print("Train Data Head:")
print(train_data.head())
print("\nTrain Data Info:")
print(train_data.info())
print("\nTrain Data Describe:")
print(train_data.describe())
print("\nMissing Values in Train Data:")
print(train_data.isnull().sum())

numeric_features = train_data.select_dtypes(include=['int64', 'float64']).columns.drop('Salary')
categorical_features = train_data.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

knn_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', KNeighborsRegressor(n_neighbors=5))
])

rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

gb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
])

svm_model = Pipeline(steps=[  # Додано SVM-класифікатор
    ('preprocessor', preprocessor),
    ('classifier', SVC(class_weight='balanced', kernel='poly', probability=True, random_state=42))
])

X_train = train_data.drop('Salary', axis=1)
y_train = train_data['Salary']

knn_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
gb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)  

X_valid = valid_data.drop('Salary', axis=1)
y_valid = valid_data['Salary']

y_pred_knn = knn_model.predict(X_valid)
y_pred_rf = rf_model.predict(X_valid)
y_pred_gb = gb_model.predict(X_valid)
y_pred_svm = svm_model.predict(X_valid)  

mape_knn = mean_absolute_percentage_error(y_valid, y_pred_knn)
mape_rf = mean_absolute_percentage_error(y_valid, y_pred_rf)
mape_gb = mean_absolute_percentage_error(y_valid, y_pred_gb)
mape_svm = mean_absolute_percentage_error(y_valid, y_pred_svm)


print(f'KNN Validation MAPE: {mape_knn:.2f}%')
print(f'Random Forest Validation MAPE: {mape_rf:.2f}%')
print(f'Gradient Boosting Validation MAPE: {mape_gb:.2f}%')
print(f'SVM Validation MAPE: {mape_svm:.2f}%')  
