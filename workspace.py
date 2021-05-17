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
import sys

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

training_data = []
validation_data = []

accuracy_dict_MNB_COUNT = {}
accuracy_dict_MNB_TFIDF = {}
accuracy_dict_GNB_COUNT = {}
accuracy_dict_GNB_TFIDF = {}
accuracy_dict_LR_COUNT = {}
accuracy_dict_LR_TFIDF = {}
accuracy_dict_RFR_COUNT = {}
accuracy_dict_RFR_TFIDF = {}

accuracy = 0
precision = 0
recall = 0

macro_avg = 0


X_train = []
y_train = []
X_val = []
y_val = []

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
        print(count)
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

def apply_models(key, processed_training_count, processed_validation_count, processed_training_tfidf, processed_validation_tfidf):
    print('Start applying models for', key, '...')
    start = time.time()

    # apply_rfr(key, processed_training_count, processed_validation_count, 100)
    # apply_rfr(key, processed_training_tfidf, processed_validation_tfidf, 100)
    
    # apply_mnb(key, processed_training_count, processed_validation_count)
    apply_mnb(key, processed_training_tfidf, processed_validation_tfidf)

    # apply_gnb(key, processed_training_count, processed_validation_count)
    # apply_gnb(key, processed_training_tfidf, processed_validation_tfidf)

    # apply_lr_cross_val(key, processed_training_count, processed_validation_count)
    # apply_lr_cross_val(key, processed_training_tfidf, processed_validation_tfidf)

    print('Done, exec time:', time.time() - start)
    # # sort results
    # accuracy_dict_GNB_TFIDF = sorted(
    #     accuracy_dict_GNB_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_GNB_COUNT = sorted(
    #     accuracy_dict_GNB_COUNT.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_MNB_TFIDF = sorted(
    #     accuracy_dict_MNB_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_MNB_COUNT = sorted(
    #     accuracy_dict_MNB_COUNT.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_LR_TFIDF = sorted(
    #     accuracy_dict_LR_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_LR_COUNT = sorted(
    #     accuracy_dict_LR_COUNT.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_RFR_TFIDF = sorted(
    #     accuracy_dict_RFR_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_RFR_COUNT = sorted(
    #     accuracy_dict_RFR_COUNT.items(), key=lambda x: x[1], reverse=True)

    # accuracy_dict_list = [accuracy_dict_GNB_TFIDF, accuracy_dict_GNB_COUNT, accuracy_dict_MNB_TFIDF,
    #      accuracy_dict_MNB_COUNT, accuracy_dict_LR_TFIDF, accuracy_dict_LR_COUNT, accuracy_dict_RFR_TFIDF, accuracy_dict_RFR_COUNT]
    # accuracy_dict_list_names = ['result_accuracy_dict_GNB_TFIDF', 'result_accuracy_dict_GNB_COUNT', 'result_accuracy_dict_MNB_TFIDF',
    #            'result_accuracy_dict_MNB_COUNT', 'result_accuracy_dict_LR_TFIDF', 'result_accuracy_dict_LR_COUNT', 'result_accuracy_dict_RFR_TFIDF', 'result_accuracy_dict_RFR_COUNT']

    # write_results(accuracy_dict_list, accuracy_dict_list_names)
    print()

def apply_rfr(key, processed_training_count, processed_validation_count, n_estimator):
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
    accuracy_dict_RFR_COUNT[str(n_estimator) + '_' + key] = score

def apply_lr_cross_val(key, training_data, validation_data):
    print('Applying LRC to', key, '...')
    LR = LogisticRegressionCV(cv=5, max_iter=100000, n_jobs=-1)
    LR.fit(training_data, y_train)

    score = LR.score(validation_data, y_val)
    accuracy_dict_LR_COUNT[key] = score

def apply_gnb(key, training_data, validation_data):
    print('Applying GNB to', key, '...')
    GNB = GaussianNB()
    GNB.fit(training_data, y_train)
    GNB.predict()
    # print ("Accuracy for G. Naive Bayes ngram: %s",    #     % (accuracy_score(y_val, GNB.predict(processed_validation_count))))
    accuracy_dict_GNB_COUNT[key] = accuracy_score(
        y_val, GNB.predict(validation_data))

def apply_mnb(key, training_data, validation_data):
    print('Applying MNB to', key, '...')
    MNB = MultinomialNB()
    MNB.fit(training_data, y_train)
    # print ("Accuracy for M. Naive Bayes ngram: %s",    #         % (accuracy_score(y_val, MNB.predict(processed_validation_count))))
    accuracy_dict_MNB_TFIDF[key] = accuracy_score(
        y_val, MNB.predict(validation_data))

# writes mutual info vocab to file and returns the vocab.
def write_and_get_mutual_info_vocab(max_features, vectorizer_type, stops, X, y):
    selected_vocab = mutual_info_feature_selection(
        stops, X, y, max_features, vectorizer_type)
    with open('mi_vocab/' + vectorizer_type + str(max_features) + '.txt', 'w') as f:
        f.write(str(selected_vocab))

    return selected_vocab

def extract_features_with_params(max_features, max_df, min_df, stops):
    feature_tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features, max_df=max_df, min_df=min_df, stop_words=stops)
    feature_count_vectorizer = {}#CountVectorizer(
        #max_features=max_features, max_df=max_df, min_df=min_df, ngram_range=(1, 2), stop_words=stops)

    processed_training_tfidf = feature_tfidf_vectorizer.fit_transform(
        X_train).toarray()
    processed_training_count = {}#feature_count_vectorizer.fit_transform(
        #X_train).toarray()

    vocab_tfidf = feature_tfidf_vectorizer.vocabulary_
    # vocab_ngram = feature_count_vectorizer.vocabulary_
    
    validation_tfidf_vectorizer = TfidfVectorizer(
        vocabulary=vocab_tfidf, stop_words=stops)
    validation_count_vectorizer = {}#CountVectorizer(
        #vocabulary=vocab_ngram, stop_words=stops)

    processed_validation_tfidf = validation_tfidf_vectorizer.fit_transform(
        X_val).toarray()
    processed_validation_count = {}#validation_count_vectorizer.fit_transform(
        #X_val).toarray()

    return processed_training_tfidf, processed_validation_tfidf, processed_training_count, processed_validation_count

def write_results(results, result_filenames):
    for i in range(len(results)):
        with open('results/' + result_filenames[i] + '.json', 'a') as f:
            w = json.dumps(results[i], indent=2)
            f.write(w)    

def test_feature_params_with_model(stops, max_features_list, min_df_list, max_df_list):
    for max_features in max_features_list:
        for max_df in max_df_list:
            for min_df in min_df_list:
                processed_training_tfidf, processed_validation_tfidf, processed_training_count, processed_validation_count = extract_features_with_params(
                    max_features, max_df, min_df, stops)
                key = str(max_df) + ' ' + str(min_df) + ' ' + str(max_features)

                apply_models(key, processed_training_count, processed_validation_count,
                             processed_training_tfidf, processed_validation_tfidf)

if __name__ == "__main__":
    args = sys.argv
    
    #folder_name = args[2]
    start = time.time()
    # # extract the training data
    # training_data = extract_data('TRAIN')
    # # extract the validation data
    # validation_data = extract_data('VAL')
    # # lemmatize the data
    # format_data(training_data)
    # format_data(validation_data)

    # for data in training_data:
    #     X_train.append(data.content)
    #     y_train.append(data.rating)
    # for data in validation_data:
    #     X_val.append(data.content)
    #     y_val.append(data.rating)

    # print('Duration:', time.time() - start)

    # write_to_txt('x_train.txt', X_train)
    # write_to_txt('y_train.txt', y_train)
    # write_to_txt('x_val.txt', X_val)
    # write_to_txt('y_val.txt', y_val)
    
    X_train = read_from_txt('x_train.txt')
    y_train = read_from_txt('y_train.txt')
    X_val = read_from_txt('x_val.txt')
    y_val = read_from_txt('y_val.txt')
    
    stops = stopwords.words('english')
    # To add more stop words, add = symbol to the beginning of the words you want to add in words.txt
    with open('words.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('='):
                stops.append(line[1:].strip())

    max_features_list = [612]
    min_df_list = [2]
    max_df_list = [x * 0.01 for x in range(30, 52)]

    # create models according to feature extraction by tf/idf
    test_feature_params_with_model(stops, max_features_list, min_df_list, max_df_list)
    sorted_acc_mnb = sorted(
        accuracy_dict_MNB_TFIDF.items(), key=lambda x: x[1], reverse=True)
    write_results([sorted_acc_mnb], ['final_mnb_tfidf_results_max_df'])
    # # create models according to mutual information selection
    # for max_features in max_features_list:
    #     selected_vocab_count = {}
    #     selected_vocab_tfidf = {}
    #     # get mutual info vocab from file if the file is created for COUNT.
    #     # else, create the vocab and save it for future use.
    #     count_filename = 'mi_vocab/count' + str(max_features) + '.txt'
    #     if os.path.exists(count_filename):
    #         print(count_filename, 'is found, retrieving vocab...')
    #         with open(count_filename, 'r') as f:
    #             s = f.read()
    #             selected_vocab_count = eval(s)
    #     else:
    #         print('Creating vocab file:', count_filename)
    #         selected_vocab_count = write_and_get_mutual_info_vocab(
    #             max_features, 'count', stops, X_train, y_train)
    #     # get mutual info vocab from file if the file is created for TFIDF.
    #     # else, create the vocab and save it for future use.
    #     tfidf_filename = 'mi_vocab/tfidf' + str(max_features) + '.txt'
    #     if os.path.exists(tfidf_filename):
    #         print(tfidf_filename, 'is found, retrieving vocab...')
    #         with open(tfidf_filename, 'r') as f:
    #             s = f.read()
    #             selected_vocab_tfidf = eval(s)
    #     else:
    #         print('Creating vocab file:', tfidf_filename)
    #         selected_vocab_tfidf = write_and_get_mutual_info_vocab(
    #             max_features, 'tfidf', stops, X_train, y_train)

    #     feature_tfidf_vectorizer = TfidfVectorizer(
    #         vocabulary=selected_vocab_tfidf, stop_words=stops)
    #     # feature_count_vectorizer = CountVectorizer(
    #     #     vocabulary=selected_vocab_count, stop_words=stops)

    #     processed_training_tfidf = feature_tfidf_vectorizer.fit_transform(
    #         X_train).toarray()
    #     processed_training_count = feature_count_vectorizer.fit_transform(
    #         X_train).toarray()

    #     processed_validation_tfidf = feature_tfidf_vectorizer.fit_transform(
    #         X_val).toarray()
    #     processed_validation_count = feature_count_vectorizer.fit_transform(
    #         X_val).toarray()
    #     # TODO(eridincu): Remind this link: https://analyticsindiamag.com/7-types-classification-algorithms/
    #     key = 'mutual info ' + str(max_features)
    #     apply_models(key, processed_training_count, processed_validation_count,
    #                  processed_training_tfidf, processed_validation_tfidf)
    # # sort results
    # accuracy_dict_GNB_TFIDF = sorted(
    #     accuracy_dict_GNB_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_GNB_COUNT = sorted(
    #     accuracy_dict_GNB_COUNT.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_MNB_TFIDF = sorted(
    #     accuracy_dict_MNB_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_MNB_COUNT = sorted(
    #     accuracy_dict_MNB_COUNT.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_LR_TFIDF = sorted(
    #     accuracy_dict_LR_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_LR_COUNT = sorted(
    #     accuracy_dict_LR_COUNT.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_RFR_TFIDF = sorted(
    #     accuracy_dict_RFR_TFIDF.items(), key=lambda x: x[1], reverse=True)
    # accuracy_dict_RFR_COUNT = sorted(
    #     accuracy_dict_RFR_COUNT.items(), key=lambda x: x[1], reverse=True)

    # accuracy_dict_list = [accuracy_dict_GNB_TFIDF, accuracy_dict_GNB_COUNT, accuracy_dict_MNB_TFIDF,
    #      accuracy_dict_MNB_COUNT, accuracy_dict_LR_TFIDF, accuracy_dict_LR_COUNT, accuracy_dict_RFR_TFIDF, accuracy_dict_RFR_COUNT]
    # accuracy_dict_list_names = ['result_accuracy_dict_GNB_TFIDF', 'result_accuracy_dict_GNB_COUNT', 'result_accuracy_dict_MNB_TFIDF',
    #            'result_accuracy_dict_MNB_COUNT', 'result_accuracy_dict_LR_TFIDF', 'result_accuracy_dict_LR_COUNT', 'result_accuracy_dict_RFR_TFIDF', 'result_accuracy_dict_RFR_COUNT']

    # write_results(accuracy_dict_list, accuracy_dict_list_names)

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
