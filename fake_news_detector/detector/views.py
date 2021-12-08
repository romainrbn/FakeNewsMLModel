from django.http import HttpResponse
from django.http import JsonResponse
from .models import Article

def index(request):
    return HttpResponse("Hello, world. You're at the detector index.")

def all(request):
    articles = Article.objects.all()
    data = [article.to_dict() for article in articles]
    return JsonResponse(data, safe=False)

def result(request):
    return HttpResponse("This is the results page. Here, the website will tell you if the article you submitted is a fake news or not, with a certainty rate.")