from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.cluster import KMeansClusterer, cosine_distance
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import json
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation as LDA

# Q1
def cluster_kmean(train_file, test_file):
    print('\nQ1:')
    train = json.load(open(train_file, 'r'))
    test = json.load(open(test_file, 'r'))
    # print(train.shape)
    # print(test)
    test_text, label = zip(*test)
    first_label = [l[0] for l in label]

    tfidf_vect = TfidfVectorizer(stop_words='english', min_df=5)
    dtm = tfidf_vect.fit_transform(train)
    test_dtm = tfidf_vect.transform(test_text)

    # print(dtm.shape)
    # print(test_dtm.shape)
    num_clusters = 3
    clusterer = KMeansClusterer(num_clusters, cosine_distance, repeats=20)
    clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)
    # print(clusters[0:5])

    predicted = [clusterer.classify(v) for v in test_dtm.toarray()]
    # print(predicted[0:10])

    df = pd.DataFrame(list(zip(first_label, predicted)), columns=['actual_class', 'cluster'])
    confusion = pd.crosstab(index=df.cluster, columns=df.actual_class)
    print('\n', confusion)

    print('\nBy using majority vote, we can find that:')
    print('cluster 0: Topic Travel & Transportation\ncluster 1: Topic Disaster and Accident\ncluster 2: Topic News and Economy')

    # Map cluster id to true labels by "majority vote"
    cluster_dict = {0: 'Travel & Transportation', 1: "Disaster and Accident", 2: 'News and Economy'}

    predicted_target = [cluster_dict[i] for i in predicted]

    print('\n', metrics.classification_report(first_label, predicted_target))


# Q2
def cluster_lda(train_file, test_file):


    train = json.load(open(train_file, 'r'))
    test = json.load(open(test_file, 'r'))
    test_text, label = zip(*test)
    first_label = [l[0] for l in label]

    tf_vectorizer = CountVectorizer(max_df=0.9, min_df=45, stop_words='english')
    tf = tf_vectorizer.fit_transform(train + list(test_text))


    # tf_feature_names = tf_vectorizer.get_feature_names()
    # print(tf_feature_names[0:10])

    num_topics = 3
    lda = LDA(n_components=num_topics, max_iter=25, verbose=1, evaluate_every=1, n_jobs=1, random_state=0).fit(tf[:len(train)])
    topic_assign = lda.transform(tf[len(train):])
    # print(topic_assign[0:10])
    cluster = topic_assign.argmax(axis=1)
    # print(cluster[0:10])

    df = pd.DataFrame(list(zip(first_label, cluster)), columns=['actual_class', 'cluster'])
    confusion = pd.crosstab(index=df.cluster, columns=df.actual_class)
    print('\n', confusion)

    print('\nBy using majority vote, we can find that:')
    print('cluster 0: Topic Travel & Transportation\ncluster 1: Topic Disaster and Accident\ncluster 2: Topic News and Economy')
    #
    # Map cluster id to true labels by "majority vote"
    cluster_dict = {0: 'Travel & Transportation', 1: "Disaster and Accident", 2: 'News and Economy'}
    predicted_target = [cluster_dict[i] for i in cluster]

    print('\n', metrics.classification_report(first_label, predicted_target))

    return topic_assign, label





# Q3
def overlapping_cluster(topic_assign, labels):
    labels = [l[0] for l in labels]
    #     print(labels)
    #     print(topic_assign)
    threshold = None
    f1 = None

    th0, th1, th2 = 0, 0, 0

    def f1C(threshold):
        thr = 0
        cluster = []
        for n in topic_assign:
            if n[0] > threshold:
                cluster.append("Travel & Transportation")
            else:
                cluster.append('NA')
        df = pd.DataFrame(list(zip(labels, cluster)), columns=['actual_class', 'cluster'])
        df = df[(df['actual_class'] == 'Travel & Transportation') | (df['cluster'] == 'Travel & Transportation')]
        #     print(df)

        actual_class = list(df['actual_class'])
        cluster = list(df['cluster'])

        h = metrics.classification_report(actual_class, cluster)

        def classification_report_csv(report):

            report_data = []
            lines = report.split('\n')
            row = []
            for line in lines:
                row_data = line.split()
                a = row_data
                #                 print(a)
                row.append(a)
            b = row[5]
            #             print(b[5])
            return (b[5])

        t = classification_report_csv(h)
        #         print(t)
        return (t)

    f10 = f1C(0.45)
    threshold0 = ('\nTravel & Transportation  0.45')

    #     print ('Travel & Transportation ', '0.45')
    #     print ('Travel & Transportation ', f10)

    def f2C(threshold):
        thr = 0
        cluster = []
        for n in topic_assign:
            if n[2] > threshold:
                cluster.append("News and Economy")
            else:
                cluster.append('NA')
        df = pd.DataFrame(list(zip(labels, cluster)), columns=['actual_class', 'cluster'])
        df = df[(df['actual_class'] == 'News and Economy') | (df['cluster'] == 'News and Economy')]
        #         print(df)

        actual_class = list(df['actual_class'])
        cluster = list(df['cluster'])

        h = metrics.classification_report(actual_class, cluster)

        def classification_report_csv(report):

            report_data = []
            lines = report.split('\n')
            row = []
            for line in lines:
                row_data = line.split()
                a = row_data
                #                 print(a)
                row.append(a)
            b = row[4]
            #             print(b[5])
            return (b[5])

        t = classification_report_csv(h)
        #         print(t)
        return (t)

    f12 = f2C(0.4)
    threshold2 = ('\nNews and Economy  0.4')

    #     print ('News and Economy ', '0.4')
    #     print ('News and Economy ', f12)

    def f1C(threshold):
        thr = 0
        cluster = []
        for n in topic_assign:
            if n[1] > threshold:
                cluster.append("Disaster and Accident")
            else:
                cluster.append('NA')
        df = pd.DataFrame(list(zip(labels, cluster)), columns=['actual_class', 'cluster'])
        df = df[(df['actual_class'] == 'Disaster and Accident') | (df['cluster'] == 'Disaster and Accident')]
        #         print(df)

        actual_class = list(df['actual_class'])
        cluster = list(df['cluster'])

        h = metrics.classification_report(actual_class, cluster)


        def classification_report_csv(report):

            report_data = []
            lines = report.split('\n')
            row = []
            for line in lines:
                row_data = line.split()
                a = row_data
                #                 print(a)
                row.append(a)
            b = row[2]
            #             print(b[5])
            return (b[5])

        t = classification_report_csv(h)
        #         print(t)
        return (t)

    f11 = f1C(0.25)
    threshold1 = ('\nDisaster and Accident  0.25')
    #     print ('Disaster and Accident ', '0.25')
    #     print ('Disaster and Accident ', f11)

    final_thresh = "Final thresholds: \n" + threshold0 + threshold1 + threshold2
    f1 = "f1-scores: \n\n" + "Travel & Transport  {}\n".format(f10) + "News and Economy  {}\n".format(
        f12) + "Disaster and Accident  {}".format(f11)

    return final_thresh, f1


if __name__ == "__main__":

    # Q1
    cluster_kmean('/Users/haodong/Desktop/train_text.json', '/Users/haodong/Desktop/test_text.json')


    # Q2
    topic_assign, labels = cluster_lda('/Users/haodong/Desktop/train_text.json', '/Users/haodong/Desktop/test_text.json')


    # Q3
    threshold, f1 = overlapping_cluster(topic_assign, labels)
    print(threshold)
    print(f1)
