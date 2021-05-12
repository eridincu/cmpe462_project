'''
References:
    - https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
'''

import os
import re
import json
import io
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer

training_data = []
validation_data = []

WNL = WordNetLemmatizer()

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

def lemmatize_sentence(sentence):
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(WNL.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

# extract the training data
for filename in os.listdir('TRAIN'):
    with io.open('TRAIN\\' + filename, "r", encoding="utf8") as f:
        review = f.read()
        
        if len(review) > 0:
            review = review.split('\n', 1)
            #header = review[0]
            content = review[0] + ' ' + review[1]
            rating = filename[-5]

            training_data.append({ 'header': 'header', 'content': content, 'rating': rating })

# extract the validation data
for filename in os.listdir('VAL'):
    with io.open('VAL\\' + filename, "r", encoding="utf8") as f:
        review = f.read()

        if len(review) > 0:
            review = review.split('\n', 1)
            #header = review[0]
            content =  review[0] + ' ' + review[1]
            rating = filename[-5]

            validation_data.append({ 'header': 'header', 'content': content, 'rating': rating })

for i in range(0, len(training_data)):
    # Remove all the special characters
    formatted_content = re.sub(r'\W', ' ', str(training_data[i]['content']))

    # remove all single characters
    formatted_content= re.sub(r'\s+[a-zA-Z]\s+', ' ', formatted_content)

    # Remove single characters from the start
    formatted_content = re.sub(r'\^[a-zA-Z]\s+', ' ', formatted_content) 

    # Substituting multiple spaces with single space
    formatted_content = re.sub(r'\s+', ' ', formatted_content, flags=re.I)

    # Removing prefixed 'b'
    formatted_content = re.sub(r'^b\s+', '', formatted_content)

    # Converting to Lowercase
    formatted_content = formatted_content.lower()

    training_data[i]['content'] = lemmatize_sentence(formatted_content)

features_sentence = list()

for data in training_data:
    features_sentence.append(data['content'])

vectorizer_1 = TfidfVectorizer (max_features=25, min_df=7, max_df=0.5, stop_words=stopwords.words('english'))
vectorizer_2 = TfidfVectorizer (max_features=25, min_df=7, max_df=0.6, stop_words=stopwords.words('english'))
vectorizer_3 = TfidfVectorizer (max_features=25, min_df=7, max_df=0.7, stop_words=stopwords.words('english'))
vectorizer_4 = TfidfVectorizer (max_features=25, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
vectorizer_5 = TfidfVectorizer (max_features=25, min_df=7, max_df=0.9, stop_words=stopwords.words('english'))

processed_features_1 = vectorizer_1.fit_transform(features_sentence).toarray()
processed_features_2 = vectorizer_2.fit_transform(features_sentence).toarray()
processed_features_3 = vectorizer_3.fit_transform(features_sentence).toarray()
processed_features_4 = vectorizer_4.fit_transform(features_sentence).toarray()
processed_features_5 = vectorizer_5.fit_transform(features_sentence).toarray()

feature_names_1 = vectorizer_1.get_feature_names()
feature_names_2 = vectorizer_2.get_feature_names()
feature_names_3 = vectorizer_3.get_feature_names()
feature_names_4 = vectorizer_4.get_feature_names()
feature_names_5 = vectorizer_5.get_feature_names()

print('feature names_1:', feature_names_1)
print('feature names_2:', feature_names_2)
print('feature names_3:', feature_names_3)
print('feature names_4:', feature_names_4)
print('feature names_5:', feature_names_5)
#print(feature_names)

#print(training_data)
#print(validation_data)
        