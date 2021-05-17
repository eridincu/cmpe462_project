import os
import re
import io
import nltk
import sys
import pickle

import numpy as np
from nltk import tokenize
from numpy.lib.function_base import vectorize

from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB


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

def extract_data(directory_path):
    data = []
    print('Extracting data...')
    sorted_filenames = os.listdir(directory_path)
    sorted_filenames.sort(key=lambda x: int(x[:-6]))
    for filename in sorted_filenames:
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

def lemmatize_sentence(sentence):
    # init lemmatizer
    WNL = WordNetLemmatizer()
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

def vectorize_X(X, vocabulary, stops):
    tfidf_vectorizer = TfidfVectorizer(
        vocabulary=vocabulary, stop_words=stops)

    return tfidf_vectorizer.fit_transform(
        X).toarray()

def get_accuracy(result, y):
    comparison = (result == y)
    comparison = comparison * 1
    accuracy = np.mean(comparison)
    
    return accuracy

def get_precision(result, y):
    correct_predictions = {'P': 0, 'N': 0, 'Z': 0}

    for i in range(len(result)):
        prediction = result[i]
        if result[i] == y[i]:
            correct_predictions[prediction] += 1
    
    for prediction_class in correct_predictions:
        total_count = len(result[result == prediction_class])
        correct_predictions[prediction_class] =  correct_predictions[prediction_class] / total_count
        
    return correct_predictions

def get_recall(result, y):
    correct_predictions = {'P': 0, 'N': 0, 'Z': 0}

    for i in range(len(result)):
        prediction = result[i]
        if result[i] == y[i]:
            correct_predictions[prediction] += 1
    
    for prediction_class in correct_predictions:
        total_count = len(y[y == prediction_class])
        correct_predictions[prediction_class] =  correct_predictions[prediction_class] / total_count
        
    return correct_predictions

def get_performance_metrics(result, y):
    result = np.array(result)
    y = np.array(y)
    accuracy = get_accuracy(result, y)
    precision = get_precision(result, y)
    recall = get_recall(result, y)
    
    macro_avg_precision = sum(precision.values()) / len(precision)
    macro_avg_recall = sum(recall.values()) / len(recall)
    return accuracy, precision, recall, macro_avg_precision, macro_avg_recall

def print_results(result):
    s = ""
    for res in result:
        s += str(res)
    print(s)
    return s
def main():
    X_test = []
    y_test = []
    
    MODEL = {}

    args = sys.argv
    # test folder path
    if len(args) <= 1:
        print('Missing arguments.')
        exit()
    
    PKL_FILENAME = args[1] # TODO(): decide on file extension
    FOLDER_NAME = args[2]

    # obtain data from the test folder
    test_data = extract_data(FOLDER_NAME)
   
    # lemmatize the data
    format_data(test_data)
   
    # create data 
    for data in test_data:
        X_test.append(data.content)
        y_test.append(data.rating)

    vocabulary = {}
    stops = {}
    with open(PKL_FILENAME, 'rb') as file:
        try:
            p = pickle.load(file)
            MODEL = p['model']
            vocabulary = p['vocabulary']
            stops = p['stops']
        except:
            print('Failed to pickle the model.')
    
    vectorized_X_test = vectorize_X(X_test, vocabulary, stops)
    
    result = MODEL.predict(vectorized_X_test)

    accuracy, precision, recall, macro_avg_precision, macro_avg_recall = get_performance_metrics(result, y_test)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('Macro Average Precision:', macro_avg_precision)
    print('Macro Average Recall:', macro_avg_recall)
    # report_statistics(accuracy, precision, recall, macro_avg_precision, macro_avg_recall)
if __name__ == "__main__":
    main()