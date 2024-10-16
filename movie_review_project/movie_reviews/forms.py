from django import forms

class MovieReviewForm(forms.Form):
    text = forms.CharField(widget=forms.Textarea, label="Введите рецензию")
