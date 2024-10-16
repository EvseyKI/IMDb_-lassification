import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score


def identity_tokenizer(text):
    return text

def identity_preprocessor(text):
    return text


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path = os.path.join(base_dir, 'data_files')
clean_train_df = pd.read_csv(os.path.join(data_path, 'clean_train_data.csv'))
clean_test_df = pd.read_csv(os.path.join(data_path, 'clean_test_data.csv'))

clean_texts_train = clean_train_df['text']
clean_texts_test = clean_test_df['text']
labels_train = clean_train_df['label']
labels_test = clean_test_df['label']

tfidf_vect = TfidfVectorizer(
    min_df=5,
    max_df=0.9,
    ngram_range=(1,2)
)

train_tfidf = tfidf_vect.fit_transform(clean_texts_train)
test_tfidf = tfidf_vect.transform(clean_texts_test)

classifiers = {
    "LinearSVC": LinearSVC(max_iter=10000)
}

def evaluate_model(train_vec, test_vec, model, vectorizer_name, model_name):
    model.fit(train_vec, labels_train)
    predictions = model.predict(test_vec)

    accuracy = accuracy_score(labels_test, predictions)
    f1_macro = f1_score(labels_test, predictions, average='macro')
    f1_weighted = f1_score(labels_test, predictions, average='weighted')

    print(f'Accuracy_score: {accuracy:.4f} ({vectorizer_name} + {model_name})')
    print(f'F1 Macro: {f1_macro:.4f}, F1 Weighted: {f1_weighted:.4f}\n')

evaluate_model(train_tfidf, test_tfidf, classifiers['LinearSVC'], 'TfidfVectorizer', 'LinearSVC')


model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

# Сохранение модели и векторизатора
joblib.dump(tfidf_vect, os.path.join(model_dir, 'tfidf_vectorizer.pkl'))
joblib.dump(classifiers["LinearSVC"], os.path.join(model_dir, 'sentiment_model.pkl'))