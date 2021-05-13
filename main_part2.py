'''
References:
    - https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258
'''

import os
import re
import json
import io
import nltk
import time
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
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


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


def write_to_txt(name, _list):
    txt_file = open(name, "w", encoding = "utf-8")
    count = 0
    for element in _list:
        txt_file.write(element + "\n")
        print(count)
        count += 1
    txt_file.close()


def read_from_txt(name):
    txt_file = open(name, "r", encoding = "utf-8")
    content_list = txt_file.readlines()
    _list = list()
    for element in content_list:
        _list.append(element.strip())
    return _list
    
    
if __name__ == "__main__":
    '''
    start = time.time()
    # extract the training data
    training_data = extract_data('TRAIN')
    # extract the validation data
    validation_data = extract_data('VAL')
    # lemmatize the data
    format_data(training_data)
    format_data(validation_data)

    for data in training_data:
        X_train.append(data.content)
        y_train.append(data.rating)
    for data in validation_data:
        X_val.append(data.content)
        y_val.append(data.rating)
    
    print('Duration:', time.time() - start)
    
    write_to_txt('x_train.txt', X_train)
    write_to_txt('y_train.txt', y_train)
    write_to_txt('x_val.txt', X_val)
    write_to_txt('y_val.txt', y_val)
    '''
    X_train = read_from_txt('x_train.txt')
    y_train = read_from_txt('y_train.txt')
    X_val = read_from_txt('x_val.txt')
    y_val = read_from_txt('y_val.txt')
    
    max_features_list = range(150, 200, 50)
    min_df_list = range(1, 2)
    max_df_list = [x * 0.1 for x in range(3, 4)]

    accuracy_dict_MNB_NGRAM = {}
    accuracy_dict_MNB_TFIDF = {}
    accuracy_dict_GNB_NGRAM = {}
    accuracy_dict_GNB_TFIDF = {}
    accuracy_dict_LR_NGRAM = {}
    accuracy_dict_LR_TFIDF = {}

    stops = stopwords.words('english')
    # To add more stop words, add = symbol to the beginning of the words you want to add in words.txt
    with open('words.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('='):
                 stops.append(line[1:].strip())
    
    for max_df in max_df_list:
        for min_df in min_df_list:
            for max_features in max_features_list:
                start = time.time()
                # TODO(eridincu): Remind Mutual Information for feature selection
                feature_tfidf_vectorizer = TfidfVectorizer (max_features=max_features, max_df=max_df, min_df=min_df, stop_words=stops)
                feature_ngram_vectorizer = CountVectorizer (max_features=max_features, max_df=max_df, min_df=min_df, ngram_range=(1,2), stop_words=stops)

                processed_features_tfidf = feature_tfidf_vectorizer.fit_transform(X_train).toarray()
                processed_features_count = feature_ngram_vectorizer.fit_transform(X_train).toarray()

                vocab_tfidf = feature_tfidf_vectorizer.vocabulary_
                vocab_ngram = feature_ngram_vectorizer.vocabulary_
                
                validation_tfidf_vectorizer = TfidfVectorizer (vocabulary=vocab_tfidf, stop_words=stopwords.words('english'))
                validation_ngram_vectorizer = CountVectorizer (vocabulary=vocab_ngram, stop_words=stopwords.words('english'))

                processed_validation_tfidf = validation_tfidf_vectorizer.fit_transform(X_val).toarray()
                processed_validation_count = validation_ngram_vectorizer.fit_transform(X_val).toarray()
                
                a = feature_ngram_vectorizer.vocabulary_
                b = validation_ngram_vectorizer.vocabulary_
                
                c = feature_tfidf_vectorizer.vocabulary_
                d = validation_tfidf_vectorizer.vocabulary_

                # TODO(eridincu): Remind this link: https://analyticsindiamag.com/7-types-classification-algorithms/
                key = str(max_df) + ' ' + str(min_df) + ' ' + str(max_features)

                MNB = MultinomialNB()
                MNB.fit(processed_features_count, y_train)
                # print ("Accuracy for M. Naive Bayes ngram: %s" 
                #         % (accuracy_score(y_val, MNB.predict(processed_validation_count))))
                accuracy_dict_MNB_NGRAM[str(max_df) + ' ' + str(min_df) + ' ' + str(max_features)] = accuracy_score(y_val, MNB.predict(processed_validation_count))

                MNB.fit(processed_features_tfidf, y_train)
                # print ("Accuracy for M. Naive Bayes tf/idf: %s" 
                #     % (accuracy_score(y_val, MNB.predict(processed_validation_tfidf))))
                accuracy_dict_MNB_TFIDF[str(max_df) + ' ' + str(min_df) + ' ' + str(max_features)] = accuracy_score(y_val, MNB.predict(processed_validation_tfidf))

                GNB = GaussianNB()
                GNB.fit(processed_features_count, y_train)
                # print ("Accuracy for G. Naive Bayes ngram: %s" 
                #     % (accuracy_score(y_val, GNB.predict(processed_validation_count))))
                accuracy_dict_GNB_NGRAM[str(max_df) + ' ' + str(min_df) + ' ' + str(max_features)] = accuracy_score(y_val, GNB.predict(processed_validation_count))

                GNB.fit(processed_features_tfidf, y_train)
                # print ("Accuracy for G. Naive Bayes tf/idf: %s" 
                #     % (accuracy_score(y_val, GNB.predict(processed_validation_tfidf))))
                accuracy_dict_GNB_TFIDF[str(max_df) + ' ' + str(min_df) + ' ' + str(max_features)] = accuracy_score(y_val, GNB.predict(processed_validation_tfidf))

                LR = LogisticRegression()
                LR.fit(processed_features_count, y_train)
                score = LR.score(processed_validation_count, y_val)
                accuracy_dict_LR_NGRAM[key] = score
                
                LR = LogisticRegression()
                LR.fit(processed_features_tfidf, y_train)
                score = LR.score(processed_validation_tfidf, y_val)
                accuracy_dict_LR_TFIDF[key] = score



    accuracy_dict_GNB_TFIDF = sorted(accuracy_dict_GNB_TFIDF.items(), key=lambda x: x[1], reverse=True)
    accuracy_dict_GNB_NGRAM = sorted(accuracy_dict_GNB_NGRAM.items(), key=lambda x: x[1], reverse=True)
    accuracy_dict_MNB_TFIDF = sorted(accuracy_dict_MNB_TFIDF.items(), key=lambda x: x[1], reverse=True)
    accuracy_dict_MNB_NGRAM = sorted(accuracy_dict_MNB_NGRAM.items(), key=lambda x: x[1], reverse=True)
    accuracy_dict_LR_TFIDF = sorted(accuracy_dict_LR_TFIDF.items(), key=lambda x: x[1], reverse=True)
    accuracy_dict_LR_NGRAM = sorted(accuracy_dict_LR_NGRAM.items(), key=lambda x: x[1], reverse=True)
    
    x = [accuracy_dict_GNB_TFIDF, accuracy_dict_GNB_NGRAM, accuracy_dict_MNB_TFIDF, accuracy_dict_MNB_NGRAM, accuracy_dict_LR_TFIDF, accuracy_dict_LR_NGRAM]
    x_names = ['accuracy_dict_GNB_TFIDF', 'accuracy_dict_GNB_NGRAM', 'accuracy_dict_MNB_TFIDF', 'accuracy_dict_MNB_NGRAM', 'accuracy_dict_LR_TFIDF', 'accuracy_dict_LR_NGRAM']

    for i in range(len(x)):
        with open('result_' + x_names[i] + '.json', 'w') as f:
            w = json.dumps(x[i], indent=2)
            f.write(w)

            



    # print('ngram', ngram_vectorizer.get_feature_names())
    # print('tfidf', feature_tfidf_vectorizer.get_feature_names())
    # print('TF/IDF Uniquesfeature_:', [x for x in feature_tfidf_vectorizer.get_feature_names() if x not in ngram_vectorizer.get_feature_names()])
    # print('NGRAM Uniques:', [x for x in feature_ngram_vectorizer.get_feature_names() if x not in feature_tfidf_vectorizer.get_feature_names()])
    # # Linear feature_SVC
    # for c in [0.001, 0.005, 0.01, 0.05, 0.1]:
    #     print('NGRAM ACCURACIES')

    #     SVM = LinearSVC(C=c, max_iter=10000)
    #     SVM.fit(processed_features_count, y_train)
    #     print ("Accuracy for C=%s: %s" 
    #         % (c, accuracy_score(y_val, SVM.predict(processed_validation_count))))
    #     print('TF/IDF ACCURACIES')
    #     SVM = LinearSVC(C=c, max_iter=10000)
    #     SVM.fit(processed_features_tfidf, y_train)
    #     print ("Accuracy for C=%s: %s" 
    #         % (c, accuracy_score(y_val, SVM.predict(processed_validation_tfidf))))
    #     print()
    
    # print('feature names_1:', feature_tfidf_vectorizer.get_feature_names())
    # print('feature names_2:', feature_ngram_vectorizer.get_feature_names())

    # print(training_data)
    # print(validation_data)
        