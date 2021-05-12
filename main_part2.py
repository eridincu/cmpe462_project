import os
import re
import json
import io

training_data = []
validation_data = []

# extract the training data
for filename in os.listdir('TRAIN'):
    with io.open('TRAIN\\' + filename, "r", encoding="utf8") as f:
        review = f.read()
        
        if len(review) > 0:
            review = review.split('\n', 1)
            header = review[0]
            content = review[1]
            rating = filename[-5]

            training_data.append({ 'header': header, 'content': content, 'rating': rating })
# extract the validation data
for filename in os.listdir('VAL'):
    with io.open('VAL\\' + filename, "r", encoding="utf8") as f:
        review = f.read()

        if len(review) > 0:
            review = review.split('\n', 1)
            header = review[0]
            content = review[1]
            rating = filename[-5]

            validation_data.append({ 'header': header, 'content': content, 'rating': rating })

print(training_data)
print(validation_data)
        