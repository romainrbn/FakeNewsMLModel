from django.db import models


# Create your models here.

class Article(models.Model):
    title = models.CharField(max_length=200)
    date = models.DateField()
    subject = models.CharField(max_length=200)
    content = models.TextField()
    is_real_news = models.BooleanField(null=True)

    def __str__(self):
        return f"Article : {self.title} -- {self.date} -- {self.subject} -- real news : {self.is_real_news}"

    def to_dict(self):
        dictionary = dict()
        dictionary['title'] = self.title
        dictionary['date'] = self.date
        dictionary['subject'] = self.subject
        dictionary['content'] = self.content
        dictionary['is_real_news'] = self.is_real_news
        return dictionary
