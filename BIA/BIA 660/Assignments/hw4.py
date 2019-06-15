# Haodong Zhao
# 10409845


import re
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
import numpy as np
from sklearn.preprocessing import normalize
import string
from scipy.spatial import distance
import matplotlib.pyplot as plt


# Q1

def extract(text):
    # use regular expression to extract data
    result = re.findall(r'(\w.*)[,] (.*) [,(].* (\d{4})[)].*[$](.*)', text)
    return result


# Q2.1

def tokenize(doc, lemmatized=False, no_stopword=False):

    # change all word to lower case
    doc = doc.lower()

    # define a pattern for word filter
    pattern = r'\w[\w\'-]*\w'

    # filter all word by using pattern
    tokens = nltk.regexp_tokenize(doc, pattern)

    # if lemmatized is True, lemmatize words
    if lemmatized:

        # pos tagging
        tagged_tokens = nltk.pos_tag(tokens)

        # define a function for pos tag
        # tag.startswith('N') is more efficient
        def getwordnet(tag):
            if tag[0] == 'N':
                return wordnet.NOUN
            elif tag[0] == 'V':
                return wordnet.VERB
            elif tag[0] == 'R':
                return wordnet.ADV
            elif tag[0] == 'J':
                return wordnet.ADJ
            else:
                return wordnet.NOUN

        # print(tagged_tokens)

        lemmatized_tokens = []

        wordnet_lemmatizer = WordNetLemmatizer()
        for word, tag in tagged_tokens:
            lemmatized_tokens.append(wordnet_lemmatizer.lemmatize(word, getwordnet(tag)))

        # update tokens
        tokens = lemmatized_tokens

    # if no_stopword is True, remove stopword
    if no_stopword:
        result = []
        stop_words = nltk.corpus.stopwords.words('english')
        for token in tokens:
            if token not in stop_words:
                result.append(token)

        # update tokens
        tokens = result

    return tokens


# Q2.2
def get_similarity(q1, q2, lemmatized=False, no_stopword=False):

    # redefine a tokeniz function for using lemmatized and no_stopword
    def tokeniz(doc):
        tokens = [token.strip() \
                  for token in nltk.word_tokenize(doc.lower()) \
                  if token.strip() not in string.punctuation]

        # if lemmatized is True, lemmatize words
        if lemmatized:

            # pos tagging
            tagged_tokens = nltk.pos_tag(tokens)

            # define a function for pos tag
            # tag.startswith('N') is more efficient
            def getwordnet(tag):
                if tag[0] == 'N':
                    return wordnet.NOUN
                elif tag[0] == 'V':
                    return wordnet.VERB
                elif tag[0] == 'R':
                    return wordnet.ADV
                elif tag[0] == 'J':
                    return wordnet.ADJ
                else:
                    return wordnet.NOUN

            # print(tagged_tokens)

            lemmatized_tokens = []

            wordnet_lemmatizer = WordNetLemmatizer()
            for word, tag in tagged_tokens:
                lemmatized_tokens.append(wordnet_lemmatizer.lemmatize(word, getwordnet(tag)))

            # update tokens
            tokens = lemmatized_tokens

        # if no_stopword is True, remove stopword
        if no_stopword:
            result = []
            stop_words = nltk.corpus.stopwords.words('english')
            for token in tokens:
                if token not in stop_words:
                    result.append(token)

            # update tokens
            tokens = result

        token_count = nltk.FreqDist(tokens)

        return token_count

    # print(len(q1))
    # print(len(q2))

    # concatenate q1 and a2
    pair = q1+q2
    # print(len(pair))
    # print(pair[3])
    # print(pair[503])

    # tokenize each question from both lists
    pair_tokens = {idx: tokeniz(doc) for idx, doc in enumerate(pair)}

    # print(pair_tokens)
    # print(len(pair_tokens))

    # get document-term matrix
    dtm = pd.DataFrame.from_dict(pair_tokens, orient='index')
    dtm = dtm.fillna(0)
    # print(dtm)

    # get normalizaed term frequency (tf) matrix
    tf = dtm.values
    doc_len = tf.sum(axis=1)
    # print(doc_len)
    tf = np.divide(tf.T, doc_len).T
    # print(tf)

    # get document frequent
    df = np.where(tf > 0, 1, 0)

    # get idf
    idf = np.log(np.divide(len(pair), np.sum(df, axis=0)))+1
    # print(idf)

    # get tf-idf
    tf_idf = normalize(tf*idf)
    # print(tf_idf)

    # find top
    # top = tf_idf.argsort()[:,::-1][:5,0:3]
    # print(top)
    # for row in top:
    #     print([dtm.columns[x] for x in row])

    # get similary matrix
    similary = 1 - distance.squareform(distance.pdist(tf_idf, 'cosine'))
    # print(similary)
    # print(len(similary))
    # sorted_similary = np.argsort(-similary)
    # print(sorted_similary)

    # find score for each pair q1 and q2
    sim = []
    for i in range(0, 500):
        j = i + 500
        score = similary[i][j]
        sim.append(score)
    # print(sim)

    return sim


# Q2.3
def predict(sim, ground_truth, threshold=0.5):

    # print(ground_truth)
    # print(sim[0])
    predict = []
    for i in range(len(sim)):
        if sim[i] > threshold:
            predict.append(1)
        else:
            predict.append(0)

    # print(predict)

    recall = 0
    is_duplicate = 0
    for i in range(len(predict)):
        if ground_truth[i] == 1:
            is_duplicate += 1
        if predict[i] == 1 and ground_truth[i] == 1:
            recall += 1

    # print(recall)
    # print(is_duplicate)
    recall /= is_duplicate
    # print(recall)

    return predict, recall


# Q3.1
def evaluate(sim, ground_truth, threshold=0.5):

    predict = []
    for i in range(len(sim)):
        if sim[i] > threshold:
            predict.append(1)
        else:
            predict.append(0)

    # print(predict)

    count_predict = 0
    count_is_duplicate = 0
    recall = 0
    precision = 0

    for i in range(len(predict)):
        if ground_truth[i] == 1:
            count_is_duplicate += 1
        if predict[i] == 1:
            count_predict += 1
        if predict[i] == 1 and ground_truth[i] == 1:
            recall += 1
            precision += 1

    # print(recall)
    # print(is_duplicate)
    precision /= count_predict
    recall /= count_is_duplicate

    return precision, recall


# Q3.2
def evaluate_update(sim, ground_truth):
    thre = 0.1
    list_threshold = []
    list_precision = []
    list_recall = []

    while thre <= 0.9:
        list_threshold.append('%.1f'%thre)
        list_precision.append(evaluate(sim, ground_truth, thre)[0])
        list_recall.append(evaluate(sim, ground_truth, thre)[1])
        thre += 0.1

    # print(list_recall)
    # print(list_precision)
    # print(list_threshold)

    # show the change in plot
    plt.title('Precision and Recall change from threshold from 0.1 to 0.9')
    plt.plot(list_threshold, list_precision, label='precision')
    plt.plot(list_threshold, list_recall, label='recall')
    plt.legend()
    plt.show()


if __name__ == "__main__":

    # Test Q1
    text='''Following is total compensation for other presidents at pr
        ivate colleges in Ohio in 2015:
        Grant Cornwell, College of Wooster (left in 2015): $911,651
        Marvin Krislov, Oberlin College (left in 2016):  $829,913
        Mark Roosevelt, Antioch College, (left in 2015): $507,672
        Laurie Joyner, Wittenberg University (left in 2015): $463,504
        Richard Giese, University of Mount Union (left in 2015): $453,800'''

    print("\nTest Q1\n")
    print(extract(text))


    data=pd.read_csv("/Users/haodong/Desktop/quora_duplicate_question_500.csv", header=0)

    q1 = data["q1"].values.tolist()
    q2 = data["q2"].values.tolist()


    # Test Q2
    print("\nTest Q2")
    print("\nlemmatized: No, no_stopword: No")
    sim = get_similarity(q1,q2)
    pred, recall = predict(sim, data["is_duplicate"].values)
    print(recall)

    print("\nlemmatized: Yes, no_stopword: No")
    sim = get_similarity(q1,q2, True)
    pred, recall = predict(sim, data["is_duplicate"].values)
    print(recall)

    print("\nlemmatized: No, no_stopword: Yes")
    sim = get_similarity(q1,q2, False, True)
    pred, recall = predict(sim, data["is_duplicate"].values)
    print(recall)

    print("\nlemmatized: Yes, no_stopword: Yes")
    sim = get_similarity(q1,q2, True, True)
    pred, recall = predict(sim, data["is_duplicate"].values)
    print(recall)


    # Test Q3. Get similarity score, set threshold, and then
    print("\nTest Q3")
    print("\nlemmatized: No, no_stopword: No")
    sim = get_similarity(q1, q2)
    prec, rec = evaluate(sim, data["is_duplicate"].values)
    print('recall is:', rec)
    print('precision is:', prec)

    print("\nlemmatized: Yes, no_stopword: No")
    sim = get_similarity(q1, q2, True)
    prec, rec = evaluate(sim, data["is_duplicate"].values)
    print('recall is:', rec)
    print('precision is:', prec)

    print("\nlemmatized: No, no_stopword: Yes")
    sim = get_similarity(q1, q2, False, True)
    prec, rec = evaluate(sim, data["is_duplicate"].values)
    print('recall is:', rec)
    print('precision is:', prec)

    print("\nlemmatized: Yes, no_stopword: Yes")
    sim = get_similarity(q1, q2, True, True)
    prec, rec = evaluate(sim, data["is_duplicate"].values)
    print('recall is:', rec)
    print('precision is:', prec)

    # test 3.2 need set different options from above, default plot will be the last option
    evaluate_update(sim, data["is_duplicate"].values)