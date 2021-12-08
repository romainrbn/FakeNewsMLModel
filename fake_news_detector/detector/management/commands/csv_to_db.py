from django.core.management.base import BaseCommand, CommandError
from datetime import datetime as datetime
from ...models import Article
import csv


def parse_date(string_date):
    months = {
        'January': 1,
        'Jan': 1,
        'February':2,
        'Feb':2,
        'March':3,
        'Mar':3,
        'April':4,
        'Apr':4,
        'May':5,
        'June':6,
        'Jun':6,
        'July':7,
        'Jul':7,
        'August':8,
        'Aug':8,
        'September':9,
        'Sep':9,
        'October':10,
        'Oct':10,
        'November':11,
        'Nov':11,
        'December':12,
        'Dec':12,
    }
    date_list = string_date.split(" ")
    year = date_list[2]
    month = months[date_list[0]]
    day = date_list[1].strip(",")
    return(f"{year}-{month}-{day}")

class Command(BaseCommand):
    help = 'Parse CSV file and dump data into database'

    def add_arguments(self, parser):
        parser.add_argument('--csv_file', nargs='+', type=str, help='The name of the CSV file, which must be in the "data" folder')
        parser.add_argument('--is_real', nargs='+', type=str, help='Tells the script if the data are real news or not')

    def handle(self, *args, **options):
        file_name = options['csv_file'][0]
        file_path = f"../data/{file_name}"
        print(options)
        is_real = False
        if(options['is_real'][0]=='True'):
            is_real = True

        with open(file_path, encoding="utf-8", newline='', mode='r') as csvfile:
            reader = csv.reader(csvfile)
            articles = []
            for index, row in enumerate(reader):
                if(len(row)==5):
                    if index != 0 and len(row[4].split(" "))==3:
                        article = Article(title=row[1], date=parse_date(row[4]), content=row[2], subject=row[3], is_real_news=is_real)
                        articles.append(article)
            Article.objects.bulk_create(articles)