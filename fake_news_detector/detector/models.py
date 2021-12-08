from django.db import models

# Create your models here.

class Article(models.Model):
    title = models.CharField(max_length=200)
    date = models.DateField()
    content = models.TextField()
    is_fake_news = models.BooleanField(null=True)

