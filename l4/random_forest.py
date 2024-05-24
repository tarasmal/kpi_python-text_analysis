from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from l4.util import read_csv_data, prepare_data_train_test_data

data = read_csv_data()
X_train, X_test, y_train, y_test = prepare_data_train_test_data(data, test_size=0.2)

vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_vectorized, y_train)
predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f"Точність моделі до використання GridSearchCV: {accuracy:.2f}")

pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', model)
])

parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'clf__n_estimators': [100, 200],
    'clf__max_depth': [10, 20, None]
}

grid_search = GridSearchCV(pipeline, parameters, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print("Точність, досягнута за допомогою покращення GridSearchCV", grid_search.score(X_test, y_test))
