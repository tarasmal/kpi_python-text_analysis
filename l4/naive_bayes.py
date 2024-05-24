from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from l4.grid_search import run_grid_search
from l4.util import read_csv_data, prepare_data_train_test_data

data = read_csv_data()
X_train, X_test, y_train, y_test = prepare_data_train_test_data(data)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
print(f"Точність моделі до використання GridSearchCV: {accuracy:.2f}")

parameters = {
    'clf__alpha': [0.01, 0.1, 1.0, 10.0]
}

best_model, best_accuracy = run_grid_search(model, parameters, X_train, y_train, X_test, y_test)

print("Точність, досягнута за допомогою покращення GridSearchCV:", best_accuracy)
