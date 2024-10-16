import os
import joblib
from django.shortcuts import render
from .forms import MovieReviewForm
from .models import MovieReview
from .utils import process_text


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

tfidf_vect = joblib.load(os.path.join(base_dir, 'movie_reviews/models/tfidf_vectorizer.pkl'))
sentiment_model = joblib.load(os.path.join(base_dir, 'movie_reviews/models/sentiment_model.pkl'))

def predict_sentiment_and_rating(review_text):
    processed_text = process_text(review_text)
    transformed_review = tfidf_vect.transform([' '.join(processed_text)])
    rating = sentiment_model.predict(transformed_review)[0]

    if rating >= 7:
        sentiment = "positive"
    else:
        sentiment = "negative"

    return rating, sentiment


def home(request):
    return render(request, 'review.html')


def submit_review(request):
    review = None

    if request.method == 'POST':
        form = MovieReviewForm(request.POST)
        if form.is_valid():
            review_text = form.cleaned_data['text']
            rating, sentiment = predict_sentiment_and_rating(review_text)
            review = MovieReview(text=review_text, rating=rating, sentiment=sentiment)
            review.save()
    else:
        form = MovieReviewForm()

    return render(request, 'review.html', {'form': form, 'review': review})


