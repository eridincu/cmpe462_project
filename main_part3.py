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

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import validation

accuracy_dict_MNB = {}
accuracy_dict_GNB = {}
accuracy_dict_LOG_REG = {}
accuracy_dict_LIN_REG = {}
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
    return lemmatized_sentence

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
        txt_file.write(" ".join(element) + "\n")
        count += 1
    txt_file.close()
    print('Done!\n')

def read_from_txt(name):
    txt_file = open(name, "r", encoding="utf-8")
    content_list = txt_file.readlines()
    _list = list()
    for element in content_list:
        _list.append(element.strip().split())
    return _list

def apply_model_tfidf(y_train, y_val, key,processed_training_tfidf, processed_validation_tfidf):
    print('Start applying models for', key, '...')
    start = time.time()

    apply_rfr(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf, 100)
    
    apply_mnb(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf)

    apply_gnb(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf)

    apply_log_reg(y_train, y_val, 'tfidf ' + key, processed_training_tfidf, processed_validation_tfidf)

    print('Done, exec time:', time.time() - start)

def apply_model_Glove(y_train, y_val, key, X_train_vector, X_val_vector):
    print('Start applying models for', key, '...')
    start = time.time()

    apply_rfr(y_train, y_val,  key, X_train_vector, X_val_vector, 1000)
    
    apply_gnb(y_train, y_val, key, X_train_vector, X_val_vector)

    apply_log_reg(y_train, y_val, key, X_train_vector, X_val_vector)
    
    apply_lin_reg(y_train, y_val, key, X_train_vector, X_val_vector)

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
def apply_log_reg(y_train, y_val, key, training_data, validation_data):
    global accuracy_dict_LOG_REG

    print('Applying LRC to', key, '...')
    LR = LogisticRegression(max_iter=100000, n_jobs=-1)
    # LR = LogisticRegressionCV(cv=5, max_iter=100000, n_jobs=-1)
    LR.fit(training_data, y_train)

    score = LR.score(validation_data, y_val)
    accuracy_dict_LOG_REG[key] = score

def apply_lin_reg(y_train, y_val, key, training_data, validation_data):
    global accuracy_dict_LIN_REG
    y_train_linreg = np.copy(y_train)
    y_val_linreg = np.copy(y_val)

    y_train_linreg[y_train_linreg == 'P'] = 3
    y_train_linreg[y_train_linreg == 'Z'] = 2
    y_train_linreg[y_train_linreg == 'N'] = 1
    y_val_linreg[y_val_linreg == 'P'] = 3
    y_val_linreg[y_val_linreg == 'Z'] = 2
    y_val_linreg[y_val_linreg == 'N'] = 1

    y_train_linreg = y_train_linreg.astype('float64')
    y_val_linreg = y_val_linreg.astype('float64')
    print('Applying LRC to', key, '...')
    LR = LinearRegression(n_jobs=-1)
    # LR = LogisticRegressionCV(cv=5, max_iter=100000, n_jobs=-1)
    LR.fit(training_data, y_train_linreg)

    score = LR.score(validation_data, y_val_linreg)
    accuracy_dict_LIN_REG[key] = score

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


def apply_svm(y_train, y_val, key, training_data, validation_data):
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    rbf_dict = {}
    for c in [0.0005, 0.5, 5, 50, 500, 5000]:
        print('Current C value:', c)
        # scaler = StandardScaler()
        # training_data = scaler.fit_transform(training_data)
        # validation_data = scaler.fit_transform(validation_data)
        SVM = svm.SVC(kernel='rbf', C=c)
        SVM.fit(training_data, y_train)
        rbf_dict[str(c)] = accuracy_score(y_val, SVM.predict(validation_data))
        print('Done!')
    print(key, c, 'RBF accuracy:', rbf_dict)
    
    linear_dict = {}
    for c in [0.0005, 0.5, 5, 50, 500, 5000]:
        print('Current C value:', c)
        SVM = svm.SVC(kernel='sigmoid', C=c)
        SVM.fit(training_data, y_train)
        print(key, 'LINEAR accuracy:', accuracy_score(y_val, SVM.predict(validation_data)))
        linear_dict[str(c)] = accuracy_score(y_val, SVM.predict(validation_data))
        print('Done!')

    polynomial_dict = {}
    for c in [0.0005, 0.5, 5, 50, 500, 5000]:
        print('Current C value:', c)
        SVM = svm.SVC(kernel='poly', C=c)
        SVM.fit(training_data, y_train)
        print(key, 'POLY accuracy:', accuracy_score(y_val, SVM.predict(validation_data)))
        polynomial_dict[str(c)] = accuracy_score(y_val, SVM.predict(validation_data))
        print('Done!')

        # print(key, c, 'POLY accuracy:', linear_dict)

    write_results([rbf_dict, linear_dict, polynomial_dict], ['rbf_gloves_new', 'sigmoid_gloves_new', 'polynomial_gloves_new'], 'results/')
    
# writes mutual info vocab to file and returns the vocab.
def write_and_get_mutual_info_vocab(max_features, vectorizer_type, stops, X, y):
    selected_vocab = mutual_info_feature_selection(
        stops, X, y, max_features, vectorizer_type)
    with open('mi_vocab/' + vectorizer_type + str(max_features) + '.txt', 'w') as f:
        f.write(str(selected_vocab))

    return selected_vocab

def extract_features_with_params(X_train, X_val, max_features, max_df, min_df, stops):
    feature_tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features, max_df=max_df, min_df=min_df, stop_words=stops)

    processed_training_tfidf = feature_tfidf_vectorizer.fit_transform(
        X_train).toarray()

    vocab_tfidf = feature_tfidf_vectorizer.vocabulary_
    
    validation_tfidf_vectorizer = TfidfVectorizer(
        vocabulary=vocab_tfidf, stop_words=stops)

    processed_validation_tfidf = validation_tfidf_vectorizer.fit_transform(
        X_val).toarray()

    return processed_training_tfidf, processed_validation_tfidf

def write_results(results, result_filenames, result_folder):
    for i in range(len(results)):
        with open(result_folder + result_filenames[i] + '.json', 'w') as f:
            if results[i] != {}:
                w = json.dumps(results[i], indent=2)
                f.write(w)    

def initialize_data(directory_name, file_name_X, file_name_y, X, y,stops):
    try:
        X = read_from_txt(file_name_X)
        y = read_from_txt(file_name_y)
    except:
        # extract the training data
        training_data = extract_data(directory_name)
        # lemmatize the data
        format_data(training_data)
        
        for data in training_data:
            content_wo_stops = [w for w in data.content if not w in stops]
            X.append(content_wo_stops)
            y.append(data.rating)

        write_to_txt(file_name_X, X)
        write_to_txt(file_name_y, y)
    
    return X, y

def getGloveVectors(total_vocabulary):  
    start = time.time()  
    print("getGloveVectors started")   
    glove = {}
    with open('glove.840B.300d.txt', 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode('utf-8')
            if word in total_vocabulary:
                vector = 100 * np.array(parts[1:], dtype=np.float32)
                glove[word] = vector
    dim = len(glove[next(iter(glove))])
    print("getGloveVectors finished in time ",time.time()-start)

    return glove, dim

def gloveVectorizer(glove, X, dimension):
    start = time.time()  
    print("gloveVectorizer started")   
    X_vector = np.array([np.mean([glove[w] for w in words if w in glove] or [np.zeros(dimension)], axis=0) for words in X])
    print("gloveVectorizer finished in time ",time.time()-start)
    return X_vector

def normalizeVector(X):
    return (X - X.min()) / (X.max() - X.min())

if __name__ == "__main__":
    training_data = []
    validation_data = []

    X_train = []
    y_train = []
    X_val = []
    y_val = []

    WNL = WordNetLemmatizer()
    
    print('Initialization begin!')
    # initialize stop words
    stops = stopwords.words('english')
    # To add more stop words, add = symbol to the beginning of the words you want to add in a file called words.txt
    with open('words.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('='):
                stops.append(line[1:].strip())
    # training data
    X_train, y_train =  initialize_data('TRAIN', 'x_train.txt', 'y_train.txt', X_train, y_train, stops)
     # validation data
    X_val, y_val = initialize_data('VAL', 'x_val.txt', 'y_val.txt', X_val, y_val, stops)
    print('Initialization done!')
    train_vocabulary = set(word for document in X_train for word in document)
    val_vocabulary = set(word for document in X_val for word in document)
    gloveDict_train, dim = getGloveVectors(train_vocabulary)
    gloveDict_val, dim = getGloveVectors(val_vocabulary)
    X_train_vector = gloveVectorizer(gloveDict_train, X_train, dim)
    X_val_vector = gloveVectorizer(gloveDict_val, X_val, dim)

    # X_train_norm = normalizeVector(X_train_vector)
    # X_val_norm = normalizeVector(X_val_vector)
   
    # apply_model_Glove(y_train, y_val, 'glove_norm', X_train_vector, X_train_vector)
    
    # apply_log_reg(y_train, y_val, "glove" , X_train_vector, X_val_vector)
    apply_svm(y_train, y_val, 'glove svm: ', X_train_vector, X_val_vector)
    # create models according to feature extraction by tf/idf
    # test_feature_params_with_model(X_train, X_val, y_train, y_val, stops)
    
    # Sort results and write them to file
    # accuracy_dict_GNB = sorted(
    #     accuracy_dict_GNB.items(), key=lambda x: x[1], reverse=True)
    
    # accuracy_dict_MNB = sorted(
    #     accuracy_dict_MNB.items(), key=lambda x: x[1], reverse=True)
    
    # accuracy_dict_LOG_REG = sorted(
    #     accuracy_dict_LOG_REG.items(), key=lambda x: x[1], reverse=True)
    
    # accuracy_dict_RFR = sorted(
    #     accuracy_dict_RFR.items(), key=lambda x: x[1], reverse=True)
    
    # prepare the data, and print the results to a file
    accuracy_dict_list = [accuracy_dict_GNB, accuracy_dict_LOG_REG, accuracy_dict_RFR, accuracy_dict_LIN_REG]
    accuracy_dict_list_names = ['result_accuracy_dict_GNB', 'result_accuracy_dict_LOG_REG', 'result_accuracy_dict_RFR','result_accuracy_dict_LIN_REG']

    write_results(accuracy_dict_list, accuracy_dict_list_names, "resultsGlove2")