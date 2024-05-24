from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


def run_grid_search(model, parameters, X_train, y_train, X_test, y_test, vectorizer=CountVectorizer(),
                    scoring='accuracy', cv=5, n_jobs=-1, verbose=1):
    pipeline = Pipeline([
        ('vect', vectorizer),
        ('clf', model)
    ])

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=n_jobs, verbose=verbose, scoring=scoring, cv=cv)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    return best_model, accuracy
