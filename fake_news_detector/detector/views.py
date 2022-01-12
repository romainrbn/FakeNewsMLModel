from django.http import HttpResponse, JsonResponse
from django.shortcuts import render
from .models import Article
from .forms import ArticleForm
import tensorflow as tf

def index(request):
    return HttpResponse("Hello, world. You're at the detector index.")

def all(request):
    articles = Article.objects.all()
    data = [article.to_dict() for article in articles]
    return JsonResponse(data, safe=False)

def inputView(request):
    # get the form

    # load the model 
    
    form = None
    # apply the model to the text from the input 
    if request.method == 'POST':
        form = ArticleForm(request.POST)
        if form.is_valid():
            # get the accuracy rate
            trained_model = tf.keras.models.load_model('/Users/romainrabouan/PycharmProjects/FakeNewsMLModel/fake_news_detector/detector/fakenewsmodel.h5')
            text_input = form.cleaned_data['article_text']
            text_input_array = [text_input]

            with open('/Users/romainrabouan/PycharmProjects/FakeNewsMLModel/fake_news_detector/detector/tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            maxlen = 1000
            text_input_array = tokenizer.texts_to_sequences(text_input_array)
            text_input_array = pad_sequences(text_input_array, maxlen=maxlen)
            print(trained_model.predict(text_input_array)[0][0])

            prediction_rate = trained_model.predict(text_input_array)[0][0]
            text_result = "La probabilité pour que cet article soit une fake news est de " + str(round((1 - prediction_rate) * 100, 3)) + "%"
            context = {"text_input" : text_result}
            #context = {"text_input" : text_input}
            return render(request, 'results.html', context=context)
    else:
        form = ArticleForm()
    
    return render(request, 'input.html', context={'form' : form})

    # return httpresponse with the rate as context


def result(request):
    return HttpResponse("This is the results page. Here, the website will tell you if the article you submitted is a fake news or not, with a certainty rate.")
