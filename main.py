from imdb import IMDb
import re
import json


# C:\Users\merdi\anaconda3\lib\site-packages\imdb\parser\http\movieParser.py
# line 1603
# paste:
#  extractor=Path('.//a[@class="title"]//text()')###

# movieparser 1634   if review.get('rating') and len(review['rating']) <= 2:
# movieparser 1635   review['rating'] = int(review['rating'])


IMDB = IMDb()

movies = IMDB.search_movie('fail', 200, False)

movies_f = IMDB.search_movie('fast', 200, False)
top_250_movies = IMDB.get_top250_movies()
indian_top_250_movies = IMDB.get_top250_indian_movies()

movies.extend(movies_f)
movies.extend(top_250_movies)
movies.extend(indian_top_250_movies)

all_reviews = dict()
final_movies = list()
movie_titles = set()
# get movies starting with letter f
for movie in movies:
    movie_data = movie.data
    movie_title = movie_data['title']
    movie_kind = movie_data['kind']
    if (bool(re.match('f', movie_title, re.I)) and movie_kind == 'movie'):
        if movie_title not in movie_titles:
            final_movies.append(movie)
            movie_titles.add(movie_title)

# LOG
print('all movies size:', len(movies))
print('final movies size:', len(final_movies))
print()
# get reviews for all movies extracted
for movie in final_movies:
    reviews = IMDB.get_movie_reviews(movie.movieID)
  
    try:
        all_reviews[movie.data['title']] = reviews['data']['reviews']
    except KeyError:
        print('No reviews available for the film:', movie)

# get total number of reviews
total_length = 0
for movie_title in all_reviews:
    total_length = total_length + len(all_reviews[movie_title])
print('total review count:', total_length)

review_count = 1
p = 0
n = 0
z = 0
unavailable = 0

for movie_title in all_reviews:
    for review in all_reviews[movie_title]:
        # create filename with review type tag
        rating = review["rating"]
        if rating is None:
            unavailable += 1
            print("One of", movie_title, "reviews doesn't have a rating")
            continue

        file_name = "F_" + str(review_count) + "_"

        review_type = ""

        if rating > 0 and rating < 4:
            if n >= 50:
                continue
            review_type = "N"
            n += 1
        elif rating >= 4 and rating < 7:
            if z >= 50:
                continue
            review_type = "Z"
            z += 1
        else:
            if p >= 50:
                continue
            review_type = "P"
            p += 1

        file_name = file_name + review_type + ".txt"
        # init file data
        title = review["title"] + "\n"
        content = review["content"]
        with open('./reviews/' + file_name, 'w') as f:
            f.write(title)
            f.write(content)
        
        review_count += 1


print('all files are created!\n')
print('positive:', p)
print('negative:', n)
print('neutral:', z)
print('unavailable:', unavailable)