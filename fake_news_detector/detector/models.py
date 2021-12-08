from django.db import models

# Create your models here.

class Article(models.Model):
    title = models.CharField(max_length=200)
    date = models.DateField()
    content = models.TextField()
    subject = models.CharField(max_length=200)
    is_real_news = models.BooleanField(null=True)

    def __str__(self):
        return f"Article : {self.title} -- {self.date} -- {self.subject} -- real news : {self.is_real_news}"