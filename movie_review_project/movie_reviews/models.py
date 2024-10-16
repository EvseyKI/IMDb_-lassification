from django.db import models

class MovieReview(models.Model):
    text = models.TextField()
    rating = models.IntegerField()
    sentiment = models.CharField(max_length=10)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Review {self.id}: {self.rating} - {self.sentiment}"
