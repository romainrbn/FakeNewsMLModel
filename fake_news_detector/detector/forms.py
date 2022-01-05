from django import forms

class ArticleForm(forms.Form):
    article_text = forms.CharField(label="Le texte de l'article que vous voulez vérifier.")