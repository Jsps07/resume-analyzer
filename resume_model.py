import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
import pickle

df = pd.read_csv("dataset.csv")

# Filled the missing values of Certifications columns
df['Certifications'] = df['Certifications'].fillna('')

# Mapping values
df['Recruiter Decision'] = df['Recruiter Decision'].map({'Hire': 1, 'Reject': 0})

def compute_strength(row):
    # Check if the candidate has certicate or not
    certificate = 1 if row['Certifications'] != '' else 0
    
    # Tokenizing Experience
    if row['Experience (Years)'] <= 2:
        exp_points = 0
    elif  2 < row['Experience (Years)'] <=5:
        exp_points = 1
    else:
        exp_points = 2

    # Project points
    if row['Projects Count'] <=3:
        proj_points = 0
    elif 4 <= row['Projects Count'] <=6:
        proj_points = 1
    else:
        proj_points = 2

    # Total Strength Score
    return certificate + exp_points + proj_points

# Create a new column
df['Strength'] = df.apply(compute_strength, axis=1)

# Checking the new column that has been created 
print(df[['Certifications', 'Experience (Years)', 'Projects Count', 'Strength']].head())
print(df['Strength'].value_counts().sort_index())
print("Min strength:", df['Strength'].min())
print("Max strength:", df['Strength'].max())

# Defining X and y
X_text = df['Skills']
X_numeric = df['Strength']
y = df['Recruiter Decision']
X = pd.concat([X_text, X_numeric], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipelines
text_pipeline = Pipeline([('tfidf', TfidfVectorizer())])

numeric_pipeline = Pipeline([('scaler', StandardScaler())])

# ColumnTransformer: column 0 = Skills, column 1 = Strength
preprocessor = ColumnTransformer([
    ('text', text_pipeline, 0),
    ('num', numeric_pipeline, [1])
])

# Full pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
])

# Train
model_pipeline.fit(X_train, y_train)

# Predict
y_pred = model_pipeline.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy: ", accuracy)
print(classification_report(y_test, y_pred))

# Saving the model
with open("resume_model.pkl", 'wb') as f:
    pickle.dump(model_pipeline, f)