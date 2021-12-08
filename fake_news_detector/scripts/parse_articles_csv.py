from ..detector.models import Article
import csv

def parse_csv_and_fill_db(csv_file, is_fake):
    with open(csv_file, newline='', mode='r') as csvfile:
        reader = csv.reader(csvfile)
        articles = []
        for row in reader:
            article = Article()
            article.title = row[0]
            article.date = row[1]
            article.content = row[2]
            article.is_fake_news = is_fake
            articles.append(article)

        Article.objects.bulk_create(articles)
