import pandas as pd
import numpy as np
import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from scipy.spatial import distance
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import scipy
import sys
import spacy
sys.setrecursionlimit(10000)

# Q1
def classify(train_file, test_file):
    # step 1
    train = pd.read_csv(train_file, header=0)
    test = pd.read_csv(test_file, header=0)
    # print(train.head())
    # print(test.shape)

    # step 2
    text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])
    parameters = {'tfidf__min_df': [1, 2, 3],
                  'tfidf__stop_words': [None, 'english'],
                  'clf__alpha': [0.5, 1, 2]}
    metric = 'f1_macro'
    gs_clf = GridSearchCV(text_clf, param_grid=parameters, scoring=metric, cv=5)
    gs_clf = gs_clf.fit(train["text"], train["label"])
    print('\nBest parameters are: ')
    for param_name in gs_clf.best_params_:
        print(param_name, ": ", gs_clf.best_params_[param_name])
    print("best f1_macro:", gs_clf.best_score_)

    # step 3
    predicted = gs_clf.predict(test["text"])
    labels = sorted(test['label'].unique())
    precision, recall, fscore, support = precision_recall_fscore_support(test['label'], predicted, labels=labels)
    print("\nPerformance:")
    print("labels: ", labels)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1-score: ", fscore)
    print("support: ", support)

    # AUC
    predict_p = gs_clf.predict_proba(test['text'])
    binary_y = np.where(test['label'] == 2, 2, 0)
    y_pred = predict_p[:, 1]
    fpr, tpr, thresholds = roc_curve(binary_y, y_pred, pos_label=2)
    print("\nAUC is:")
    print(auc(fpr, tpr))

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC of Naive Bayes Model')

    precision, recall, thresholds = precision_recall_curve(binary_y, y_pred, pos_label=2)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision_Recall_Curve of Naive Bayes Model')
    plt.show()

# Q2
def impact_of_sample_size(train_file):
    data = pd.read_csv(train_file, header=0)
    # print(data.shape)
    # print(len(data))
    # print(data['text'])
    # print(data['label'].unique())

    n = 800
    MNB_average_f1_macro = []
    LSVM_average_f1_macro = []
    MNB_average_AUC = []
    LSVM_average_AUC = []
    sample = []
    for i in range(0, 19):
        # print(n)
        dat = data[:n+1]
        tfidf_vect = TfidfVectorizer(stop_words='english')
        dtm = tfidf_vect.fit_transform(dat['text'])
        metrics = ['f1_macro', 'roc_auc']

        # MNB
        clf = MultinomialNB()
        cv = cross_validate(clf, dtm, dat['label'], scoring=metrics, cv=5, return_train_score=True)
        f1_macro = (cv["test_f1_macro"][0] + cv["test_f1_macro"][1] + cv["test_f1_macro"][2] + cv["test_f1_macro"][3] + cv["test_f1_macro"][4]) / 5
        MNB_average_f1_macro.append(f1_macro)
        auc = (cv["test_roc_auc"][0] + cv["test_roc_auc"][1] + cv["test_roc_auc"][2] + cv["test_roc_auc"][3] + cv["test_roc_auc"][4]) / 5
        MNB_average_AUC.append(auc)

        # LSVM
        clf1 = svm.LinearSVC()
        cv1 = cross_validate(clf1, dtm, dat['label'], scoring=metrics, cv=5)
        f1_macro1 = (cv1["test_f1_macro"][0] + cv1["test_f1_macro"][1] + cv1["test_f1_macro"][2] + cv1["test_f1_macro"][3] + cv1["test_f1_macro"][4]) / 5
        LSVM_average_f1_macro.append(f1_macro1)
        auc1 = (cv1["test_roc_auc"][0] + cv1["test_roc_auc"][1] + cv1["test_roc_auc"][2] + cv1["test_roc_auc"][3] + cv1["test_roc_auc"][4]) / 5
        LSVM_average_AUC.append(auc1)

        sample.append(n)
        n += 400

    # print(MNB_average_f1_macro)
    # print(LSVM_average_f1_macro)
    # print(MNB_average_AUC)
    # print(LSVM_average_AUC)

    plt.figure()
    plt.plot(sample, MNB_average_f1_macro, color='darkorange', lw=2, label='f1_MNB')
    plt.plot(sample, LSVM_average_f1_macro, color='blue', lw=2, label='f1_SVM')
    plt.legend(loc='best')
    plt.xlim([800, 8000])
    plt.xlabel('Number of sample')
    plt.ylabel('f1_score')
    plt.title('Relationship between sample size and F1-score')
    # plt.show()

    plt.figure()
    plt.plot(sample, MNB_average_AUC, color='darkorange', lw=2, label='AUC_MNB')
    plt.plot(sample, LSVM_average_AUC, color='blue', lw=2, label='AUC_SVM')
    plt.legend(loc='best')
    plt.xlim([800, 8000])
    plt.xlabel('Number of sample')
    plt.ylabel('AUC')
    plt.title('Relationship between sample size and AUC')
    plt.show()



# Q3
def classify_duplicate(filename):
    auc = None

    data = pd.read_csv(filename)
    # print(data.head())
    # print(data.shape)

    q1 = data['q1']
    q2 = data['q2']


    def tokenize(doc):
        tokens = [token.strip() for token in nltk.word_tokenize(doc.lower()) if token.strip() not in string.punctuation]
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

        result = []
        stop_words = nltk.corpus.stopwords.words('english')
        for token in tokens:
            if token not in stop_words:
                result.append(token)

        # update tokens
        tokens = result


        return tokens

    unigram = []
    bigram = []
    trigram = []
    qq1 = []
    qq2 = []

    for i in range(0,500):
        uni_n = 0
        bi_n = 0
        tri_n = 0
        uni_q1 = tokenize(q1[i])
        uni_q2 = tokenize(q2[i])
        for token in uni_q2:
            if token in uni_q1:
                uni_n += 1
        unigram.append(uni_n)

        bi_q1 = list(nltk.bigrams(uni_q1))
        bi_q2 = list(nltk.bigrams(uni_q2))
        for t in bi_q2:
            if t in bi_q1:
                bi_n += 1
        bigram.append(bi_n)

        tri_q1 = list(nltk.trigrams((uni_q1)))
        tri_q2 = list(nltk.trigrams((uni_q2)))
        for tok in tri_q2:
            if tok in tri_q1:
                tri_n += 1
        trigram.append(tri_n)
        qq1.append(uni_q1)
        qq2.append(uni_q2)


    # print(unigram)
    data['unigram'] = unigram
    data['bigram'] = bigram
    data['trigram'] = trigram
    # print(data.head())


    def tfidf(docs):

        docs_tokens = {idx: nltk.FreqDist(tokenize(doc)) \
                       for idx, doc in enumerate(docs)}

        dtm = pd.DataFrame.from_dict(docs_tokens, orient="index")
        dtm = dtm.fillna(0)

        tf = dtm.values
        doc_len = tf.sum(axis=1)
        tf = np.divide(tf, doc_len[:, None])

        df = np.sum(np.where(tf > 0, 1, 0), axis=0)

        smoothed_idf = np.log(np.divide(len(docs) + 1, df + 1)) + 1

        tf_idf = normalize(tf * smoothed_idf)

        smoothed_tf_idf = normalize(tf * smoothed_idf)

        return smoothed_tf_idf


    def get_similarity(q1, q2):
        all_q = q1 + q2
        tf_idf = tfidf(all_q)
         # cosine similarity of each row
        sim = np.sum(tf_idf[0:len(q1)] * tf_idf[len(q1):], axis=1)

        return sim

    q1 = data["q1"].values.tolist()
    q2 = data["q2"].values.tolist()
    sim = get_similarity(q1, q2)

    def predict(sim, ground_truth, threshold=0.5):

        predict = (sim > threshold).astype(int)
        pre = predict

        per = sum((predict == ground_truth) & (predict == 1)) / sum(ground_truth)

        return per

    pred = predict(sim, data['is_duplicate'].values)
    AUC = pred
    # print(pred)
    # # print(pred)
    data['pred'] = pred
    dat = data
    X = dat.iloc[:, 3:]
    y = dat.iloc[:, 2]
    # print(dat)

    cv = StratifiedKFold(n_splits=5)

    cv.get_n_splits(X, y)
    clf = svm.SVC(kernel='linear', probability=True)

    # print(cv.split(X, y))
    Auc = 0
    for i, j in cv.split(X, y):
        # print(i)
        # print(X.iloc[i].shape)
        # print(y.iloc[i].shape)
        # print(X.iloc[j].shape)
        # print(y.iloc[j].shape)
        clf = clf.fit(X.iloc[i], y.iloc[i])
        predict_p = clf.predict_proba(X.iloc[j])
        binary_y = np.where(y.iloc[j] == 1, 1, 0)
        y_pred = predict_p[:, 1]
        fpr, tpr, thresholds = roc_curve(binary_y, y_pred, pos_label=1)
        # print(auc(fpr, tpr))

    return AUC


if __name__ == "__main__":
    # # Test Q1
    # classify("/Users/haodong/Desktop/train.csv", "/Users/haodong/Desktop/test.csv")

    # # Test Q2
    # impact_of_sample_size("/Users/haodong/Desktop/train_large.csv")

    # Test Q3
    result = classify_duplicate("/Users/haodong/Desktop/quora_duplicate_question_500.csv")
    print("Q3: ", result)
