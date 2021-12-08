from django.shortcuts import render

from django.http import HttpResponse


def index(request):
    return HttpResponse("Hello, world. You're at the detector index.")

def result(request):
    return HttpResponse("This is the results page. Here, the website will tell you if the article you submitted is a fake news or not, with a certainty rate.")