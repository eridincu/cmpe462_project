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
import pickle

import numpy as np
from nltk import tokenize
from numpy.lib.function_base import vectorize

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_selection import mutual_info_classif

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import validation

accuracy_dict_MNB = {}
accuracy_dict_GNB = {}
accuracy_dict_LR = {}
accuracy_dict_RFR = {}

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
    # tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
    # tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (
        x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(WNL.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)

def extract_data(directory_path):
    data = []
    print('Extracting data...')
    filenames = os.listdir(directory_path)
    filenames.sort(key=lambda x: int(x[:-6]))
    for filename in filenames:
        with io.open(directory_path + '\\' + filename, "r", encoding="utf8") as f:
            review = f.read()

            if len(review) > 0:
                review = review.split('\n', 1)
                #header = review[0]
                content = review[0] + ' ' + review[1]
                rating = filename[-5]

                data.append(Data('header', content, rating))
    print('Done!\n')
    return data

def format_data(data):
    print('Formatting data...')
    # lemmatize the training data
    for i in range(0, len(data)):
        # Remove all the special characters
        formatted_content = re.sub(r'\W', ' ', str(data[i].content))

        # remove all single characters
        formatted_content = re.sub(r'\s+[a-zA-Z]\s+', ' ', formatted_content)

        # Remove single characters from the start
        formatted_content = re.sub(r'\^[a-zA-Z]\s+', ' ', formatted_content)

        # Remove numbers
        # formatted_content = re.sub(r'\b\d+?\b', ' ', formatted_content)
        
        # Remove links
        formatted_content = re.sub("http(s){0,1}://[\w/.]+", " ", formatted_content)

        # Substituting multiple spaces with single space
        formatted_content = re.sub(r'\s+', ' ', formatted_content, flags=re.I)

        # Removing prefixed 'b'
        formatted_content = re.sub(r'^b\s+', '', formatted_content)

        # Converting to Lowercase
        formatted_content = formatted_content.lower()

        data[i].content = lemmatize_sentence(formatted_content)
    print('Done!\n')

def write_to_txt(name, _list):
    print('Saving formatted data to txt...')
    txt_file = open(name, "w", encoding="utf-8")
    count = 0
    for element in _list:
        txt_file.write(element + "\n")
        count += 1
    txt_file.close()
    print('Done!\n')

def read_from_txt(name):
    txt_file = open(name, "r", encoding="utf-8")
    content_list = txt_file.readlines()
    _list = list()
    for element in content_list:
        _list.append(element.strip())
    return _list

def apply_models(y_train, y_val, key, processed_training_count, processed_validation_count, processed_training_tfidf, processed_validation_tfidf):
    print('Start applying models for', key, '...')
    start = time.time()

    apply_rfr(y_train, y_val, 'count ' + key, processed_training_count, processed_validation_count, 100)
    apply_rfr(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf, 100)
    
    apply_mnb(y_train, y_val, 'count ' + key, processed_training_count, processed_validation_count)
    apply_mnb(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf)

    apply_gnb(y_train, y_val, 'count ' + key, processed_training_count, processed_validation_count)
    apply_gnb(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf)

    apply_lr_cross_val(y_train, y_val, 'count ' + key, processed_training_count, processed_validation_count)
    apply_lr_cross_val(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf)

    print('Done, exec time:', time.time() - start)

def apply_rfr(y_train, y_val, key, processed_training_count, processed_validation_count, n_estimator):
    global accuracy_dict_RFR

    print('Applying RFR to', key, '...')
    y_train_rfr = np.copy(y_train)
    y_val_rfr = np.copy(y_val)

    y_train_rfr[y_train_rfr == 'P'] = 3
    y_train_rfr[y_train_rfr == 'Z'] = 2
    y_train_rfr[y_train_rfr == 'N'] = 1
    y_val_rfr[y_val_rfr == 'P'] = 3
    y_val_rfr[y_val_rfr == 'Z'] = 2
    y_val_rfr[y_val_rfr == 'N'] = 1

    y_train_rfr = y_train_rfr.astype('float64')
    y_val_rfr = y_val_rfr.astype('float64')

    RFR = RandomForestRegressor(n_estimators=n_estimator, n_jobs=-1)
    RFR.fit(processed_training_count, y_train_rfr)
    score = RFR.score(processed_validation_count, y_val_rfr)
    accuracy_dict_RFR[str(n_estimator) + '_' + key] = score

# apply logistic regression with 5-fold cross validation and save the accuracy in logistic regression result dictionary
def apply_lr_cross_val(y_train, y_val, key, training_data, validation_data):
    global accuracy_dict_LR

    print('Applying LRC to', key, '...')
    LR = LogisticRegressionCV(cv=5, max_iter=100000, n_jobs=-1)
    LR.fit(training_data, y_train)

    score = LR.score(validation_data, y_val)
    accuracy_dict_LR[key] = score

# apply logistic regression with gaussian naive bayes and save the accuracy in logistic regression result dictionary
def apply_gnb(y_train, y_val, key, training_data, validation_data):
    global accuracy_dict_GNB

    print('Applying GNB to', key, '...')
    GNB = GaussianNB()
    GNB.fit(training_data, y_train)
    
    accuracy_dict_GNB[key] = accuracy_score(
        y_val, GNB.predict(validation_data))

# apply logistic regression with gaussian naive bayes and save the accuracy in logistic regression result dictionary
def apply_mnb(y_train, y_val, key, training_data, validation_data):
    global accuracy_dict_MNB

    print('Applying MNB to', key, '...')
    MNB = MultinomialNB()
    MNB.fit(training_data, y_train)
    # print ("Accuracy for M. Naive Bayes ngram: %s",    #         % (accuracy_score(y_val, MNB.predict(processed_validation_count))))
    accuracy_dict_MNB[key] = accuracy_score(
        y_val, MNB.predict(validation_data))

# writes mutual info vocab to file and returns the vocab.
def write_and_get_mutual_info_vocab(max_features, vectorizer_type, stops, X, y):
    selected_vocab = mutual_info_feature_selection(
        stops, X, y, max_features, vectorizer_type)
    with open('mi_vocab/' + vectorizer_type + str(max_features) + '.txt', 'w') as f:
        f.write(str(selected_vocab))

    return selected_vocab

def mutual_info_feature_selection(stops, X, y, threshold, vectorizer_type):
    vectorizer = ""

    if vectorizer_type == 'tfidf':
        vectorizer = TfidfVectorizer(stop_words=stops)
    else:
        vectorizer = CountVectorizer(stop_words=stops)
    print('init vectorizer')

    features = vectorizer.fit_transform(X).toarray()
    print('create features')
    feature_scores = mutual_info_classif(features, y, random_state=0)
    print('received feature scores')
    high_score_features = {}
    count = 0
    for score, f_name in sorted(zip(feature_scores, vectorizer.get_feature_names()), reverse=True)[:threshold]:
        high_score_features[f_name] = count
        count += 1
    print('sort feature scores')
    return high_score_features
    
def extract_features_with_params(X_train, X_val, max_features, max_df, min_df, stops):
    feature_tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features, max_df=max_df, min_df=min_df, stop_words=stops)
    feature_count_vectorizer = CountVectorizer(
        max_features=max_features, max_df=max_df, min_df=min_df, ngram_range=(1, 2), stop_words=stops)

    processed_training_tfidf = feature_tfidf_vectorizer.fit_transform(
        X_train).toarray()
    processed_training_count = feature_count_vectorizer.fit_transform(
        X_train).toarray()

    vocab_tfidf = feature_tfidf_vectorizer.vocabulary_
    vocab_ngram = feature_count_vectorizer.vocabulary_
    
    validation_tfidf_vectorizer = TfidfVectorizer(
        vocabulary=vocab_tfidf, stop_words=stops)
    validation_count_vectorizer = CountVectorizer(
        vocabulary=vocab_ngram, stop_words=stops)

    processed_validation_tfidf = validation_tfidf_vectorizer.fit_transform(
        X_val).toarray()
    processed_validation_count = validation_count_vectorizer.fit_transform(
        X_val).toarray()

    return processed_training_tfidf, processed_validation_tfidf, processed_training_count, processed_validation_count

def write_results(results, result_filenames):
    for i in range(len(results)):
        with open('results/' + result_filenames[i] + '.json', 'w') as f:
            w = json.dumps(results[i], indent=2)
            f.write(w)    

def test_feature_params_with_model(X_train, X_val, y_train, y_val, stops, max_features_list, min_df_list, max_df_list):
    for max_features in max_features_list:
        for max_df in max_df_list:
            for min_df in min_df_list:
                processed_training_tfidf, processed_validation_tfidf, processed_training_count, processed_validation_count = extract_features_with_params(
                    X_train, X_val, max_features, max_df, min_df, stops)
                key = str(max_df) + ' ' + str(min_df) + ' ' + str(max_features)

                apply_models(y_train, y_val, key, processed_training_count, processed_validation_count,
                             processed_training_tfidf, processed_validation_tfidf)

def vectorize_data_tfidf_and_count(tfidf_vectorizer, count_vectorizer, X):
    tfidf_data = tfidf_vectorizer.fit_transform(
            X).toarray()
    count_data = count_vectorizer.fit_transform(
            X).toarray()

    return tfidf_data, count_data

def select_mutual_info_vocab(X_train, y_train, stops, vectorizer_type, max_features, tfidf_filename):
    selected_vocab = {}
    try:
        with open(tfidf_filename, 'r') as f:
            s = f.read()
            selected_vocab = eval(s)
    except:
        print(tfidf_filename, 'is not found. Creating vocab file:', tfidf_filename)
        selected_vocab = write_and_get_mutual_info_vocab(
            max_features, vectorizer_type, stops, X_train, y_train)
    return selected_vocab

def test_feature_params_with_model_mutual_info(X_train, X_val, y_train, y_val, stops, max_features_list):
    for max_features in max_features_list:
        # get mutual info vocab from file if the file is created for COUNT.
        # else, create the vocab and save it for future use.
        count_filename = 'mi_vocab/count' + str(max_features) + '.txt'
        selected_vocab_count =  select_mutual_info_vocab(X_train, y_train, stops, 'count', max_features, count_filename)
        # get mutual info vocab from file if the file is created for TFIDF.,        # else, create the vocab and save it for future use.
        tfidf_filename = 'mi_vocab/tfidf' + str(max_features) + '.txt'
        selected_vocab_tfidf = select_mutual_info_vocab(X_train, y_train, stops, 'tfidf', max_features, tfidf_filename)
       
        # vectorize X data according to tf-idf and count
        feature_tfidf_vectorizer = TfidfVectorizer(
            vocabulary=selected_vocab_tfidf, stop_words=stops)
        feature_count_vectorizer = CountVectorizer(
            vocabulary=selected_vocab_count, stop_words=stops)

        # create vectorized input data
        processed_training_tfidf, processed_training_count = vectorize_data_tfidf_and_count(feature_tfidf_vectorizer, feature_count_vectorizer, X_train)
        processed_validation_tfidf, processed_validation_count = vectorize_data_tfidf_and_count(feature_tfidf_vectorizer, feature_count_vectorizer, X_val)
        
        # Apply models
        key = 'mutual info ' + str(max_features)
        apply_models(y_train, y_val, key, processed_training_count, processed_validation_count,
                     processed_training_tfidf, processed_validation_tfidf)

def initialize_data(directory_name, file_name_X, file_name_y, X, y):
    try:
        X = read_from_txt(file_name_X)
        y = read_from_txt(file_name_y)
    except:
        # extract the training data
        training_data = extract_data(directory_name)
        # lemmatize the data
        format_data(training_data)
        
        for data in training_data:
            X.append(data.content)
            y.append(data.rating)

        write_to_txt(file_name_X, X)
        write_to_txt(file_name_y, y)
    
    return X, y

if __name__ == "__main__":
    training_data = []
    validation_data = []

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    WNL = WordNetLemmatizer()
    
    print('Initialization begin!')
    # training data
    X_train, y_train =  initialize_data('TRAIN', 'x_train.txt', 'y_train.txt', X_train, y_train)
     # validation data
    X_val, y_val = initialize_data('VAL', 'x_val.txt', 'y_val.txt', X_val, y_val)
    print('Initialization done!')
    
    # initialize stop words
    stops = stopwords.words('english')
    # To add more stop words, add = symbol to the beginning of the words you want to add in a file called words.txt
    with open('words.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('='):
                stops.append(line[1:].strip())

    max_features_list = range(1, 750, 1)
    min_df_list = range(2, 3)
    max_df_list = [x * 0.1 for x in range(1, 11)]

    # create models according to feature extraction by tf/idf
    test_feature_params_with_model(X_train, X_val, y_train, y_val, stops, max_features_list, min_df_list, max_df_list)

    # create models according to mutual information selection
    test_feature_params_with_model_mutual_info(X_train, X_val, y_train, y_val, stops, max_features_list)
    
    # Sort results and write them to file
    accuracy_dict_GNB = sorted(
        accuracy_dict_GNB.items(), key=lambda x: x[1], reverse=True)
    
    accuracy_dict_MNB = sorted(
        accuracy_dict_MNB.items(), key=lambda x: x[1], reverse=True)
    
    accuracy_dict_LR = sorted(
        accuracy_dict_LR.items(), key=lambda x: x[1], reverse=True)
    
    accuracy_dict_RFR = sorted(
        accuracy_dict_RFR.items(), key=lambda x: x[1], reverse=True)
    
    # prepare the data, and print the results to a file
    accuracy_dict_list = [accuracy_dict_GNB, accuracy_dict_MNB, accuracy_dict_LR, accuracy_dict_RFR]
    accuracy_dict_list_names = ['result_accuracy_dict_GNB', 'result_accuracy_dict_MNB', 'result_accuracy_dict_LR', 'result_accuracy_dict_RFR']

    write_results(accuracy_dict_list, accuracy_dict_list_names)