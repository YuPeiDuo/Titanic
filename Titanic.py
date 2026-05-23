import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

df = pd.read_csv("D:/Machine_learning_Kaggle/train.csv")

df_clean = df.copy()
df_clean['Title'] = df_clean['Name'].apply(lambda x: x.split(',')[1].split(',')[0].strip())
df_clean['Age'].fillna(df_clean['Age'].median(),inplace=True)
df_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0],inplace=True)

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Title']
X = df_clean[features]
y = df_clean['Survived']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

numeric_features = ['Age','SibSp','Parch','Fare']
categorical_features = ['Pclass','Sex','Embarked','Title']

numeric_transformer = Pipeline([('scaler',StandardScaler())])
categorical_transformer = Pipeline([('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num',numeric_transformer,numeric_features),
    ('cat',categorical_transformer,categorical_features)
])

pipeline = Pipeline([('preprocessor',preprocessor),('classifier',RandomForestClassifier(random_state=42))])

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_val)

param_grid = {
    'classifier__n_estimators': [50,100,150],
    'classifier__max_depth': [5,10,None],
    'classifier__min_samples_split': [2,5,10]
}
grid = GridSearchCV(pipeline,param_grid,cv=5,scoring='accuracy',n_jobs=-1)
grid.fit(X_train,y_train)

best = grid.best_estimator_
y_pred_best = best.predict(X_val)

test_df = pd.read_csv("D:/Machine_learning_Kaggle/test.csv")

test_clean = test_df.copy()

test_clean['Title'] = test_clean['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
test_clean['Title'] = test_clean['Title'].replace(['Mlle','Ms','Mme'], 'Miss')
test_clean['Title'] = test_clean['Title'].replace(['Lady','Countess','Dona','Jonkheer','Sir','Don'], 'Royalty')
test_clean['Title'] = test_clean['Title'].replace(['Capt','Col','Major','Dr','Rev'], 'Officer')

test_clean['Age'].fillna(df_clean['Age'].median(), inplace=True)
test_clean['Fare'].fillna(df_clean['Fare'].median(), inplace=True)
test_clean['Embarked'].fillna(df_clean['Embarked'].mode()[0], inplace=True)

X_test = test_clean[features]

y_pred_test = best.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': y_pred_test
})

submission.to_csv('D:/Machine_learning_Kaggle/submission.csv',index=False)