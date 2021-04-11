from imdb import IMDb
import re
import json

# C:\Users\merdi\anaconda3\lib\site-packages\imdb\parser\http\__init__.py -> 
# line 539
# paste:
#  return {'review': self.mProxy.reviews_parser.parse(cont), 'review_base': cont}

# C:\Users\merdi\anaconda3\lib\site-packages\imdb\parser\http\movieParser.py
# line 1603
# paste:
#  extractor=Path('.//a[@class="title"]//text()')###

IMDB = IMDb()

movies = IMDB.search_movie('f', 200, False)
top_250_movies = IMDB.get_top250_movies()
indian_top_250_movies = IMDB.get_top250_indian_movies()

all_reviews = {}

print('all movies size:', len(movies))

final_movies = []

# get movies starting with letter f
for movie in movies:
    movie_data = movie.data
    movie_title = movie_data['title']
    movie_kind = movie_data['kind']
    if (bool(re.match('f', movie_title, re.I)) and movie_kind == 'movie'):
        final_movies.append(movie)

for top_movie in top_250_movies:
    movie_data = top_movie.data
    movie_title = movie_data['title']
    movie_kind = movie_data['kind']
    if (bool(re.match('f', movie_title, re.I)) and movie_kind == 'movie'):
        final_movies.append(top_movie)

for indian_top_movie in indian_top_250_movies:
    indian_top_movie_data = indian_top_movie.data
    movie_title = movie_data['title']
    movie_kind = movie_data['kind']
    if (bool(re.match('f', movie_title, re.I)) and movie_kind == 'movie'):
        final_movies.append(indian_top_movie)

print('length of final movies:', len(final_movies))

# get reviews for all movies extracted
for movie in final_movies:
    reviews = IMDB.get_movie_reviews(movie.movieID)
    # for debugging
    cont = reviews['review_base']
    reviews = reviews['review']
    try:
        all_reviews[movie.data['title']] = reviews['data']['reviews']
    except KeyError:
        print('No reviews available for the film:', movie)

# get total number of reviews
total_length = 0
for movie_title in all_reviews:
    total_length = total_length + len(all_reviews[movie_title])
print('total review count:', total_length)

# dump to json file
with open('reviews.json', 'w') as f:
    json_reviews = json.dumps(all_reviews)
    f.write(json_reviews)