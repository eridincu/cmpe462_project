'''
References:
    - https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
'''

import os
import re
import json
import io
import nltk
from nltk import tokenize
from numpy.lib.function_base import vectorize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

training_data = []
validation_data = []

WNL = WordNetLemmatizer()

class Data():
    def __init__(self, header, content, rating):
        self.header = header
        self.content = content
        self.rating = rating

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

def extract_data(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        with io.open(directory_path + '\\' + filename, "r", encoding="utf8") as f:
            review = f.read()
            
            if len(review) > 0:
                review = review.split('\n', 1)
                #header = review[0]
                content = review[0] + ' ' + review[1]
                rating = filename[-5]

                data.append(Data('header', content, rating))
    return data

def format_data(data):
    # lemmatize the training data
    for i in range(0, len(data)):
        # Remove all the special characters
        formatted_content = re.sub(r'\W', ' ', str(data[i].content))

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

        data[i].content = lemmatize_sentence(formatted_content)

# extract the training data
training_data = extract_data('TRAIN')
# extract the validation data
validation_data = extract_data('VAL')
# lemmatize the data
format_data(training_data)
format_data(validation_data)

X_train = list()
y_train = list()
X_val = list()
y_val = list()

for data in training_data:
    X_train.append(data.content)
    y_train.append(data.rating)
for data in validation_data:
    X_val.append(data.content)
    y_val.append(data.rating)

tfidf_vectorizer = TfidfVectorizer (max_features=50, min_df=7, max_df=0.6, stop_words=stopwords.words('english'))
ngram_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=7, max_df=0.6, max_features=50, stop_words=stopwords.words('english'))

processed_features_tfidf = tfidf_vectorizer.fit_transform(X_train).toarray()
processed_features_count = ngram_vectorizer.fit_transform(X_train).toarray()

processed_validation_tfidf = tfidf_vectorizer.fit_transform(X_val).toarray()
processed_validation_count = ngram_vectorizer.fit_transform(X_val).toarray()

# Linear SVC
for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    print('NGRAM ACCURACIES')
    SVM = LinearSVC(C=c, max_iter=10000)
    SVM.fit(processed_features_count, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, SVM.predict(processed_validation_count))))
    print('TF/IDF ACCURACIES')
    SVM = LinearSVC(C=c, max_iter=10000)
    SVM.fit(processed_features_tfidf, y_train)
    print ("Accuracy for C=%s: %s" 
           % (c, accuracy_score(y_val, SVM.predict(processed_validation_tfidf))))
    print()

MNB = MultinomialNB()
MNB.fit(processed_features_count, y_train)
print ("Accuracy for Naive Bayes ngram: %s" 
        % (accuracy_score(y_val, MNB.predict(processed_validation_count))))
MNB.fit(processed_features_tfidf, y_train)
print ("Accuracy for Naive Bayes tf/idf: %s" 
        % (accuracy_score(y_val, MNB.predict(processed_validation_tfidf))))

#print('feature names_1:', tfidf_vectorizer.get_feature_names())
#print('feature names_2:', ngram_vectorizer.get_feature_names())
#print(feature_names)

#print(training_data)
#print(validation_data)
        