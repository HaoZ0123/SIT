# Structure of your solution to Assignment 1
import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain


# Q1
#
def analyze_tf_idf(arr, K):
    # get length of each document
    length_of_documents = np.sum(arr, axis=1)

    # normalizes the frequency of each word
    tf = arr.T / length_of_documents
    tf = tf.T

    # calculates the document frequency of each word
    df = np.sum(tf, axis=0)

    # calculates tf_idf
    tf_idf = tf / (np.log(df) + 1)

    # sort words in each document from greatest to smallest and present the words' original index
    a = np.argsort(tf_idf)[:, ::-1]

    # sort words in each document from greatest to smallest and present the words' frequency
    b = np.sort(tf_idf)[:, ::-1]

    # slice the top K frequency words and present their original index
    top_k = a[:, :K]
    return tf_idf, top_k


# Q2
#
def analyze_data(filepath):
    # import csv file as a dataframe
    data = pd.DataFrame(pd.read_csv(filepath, header=0))

    # select data which answercount > 0
    dat1 = data[data.answercount > 0]

    # sort data by viewcount from greatest to smallest
    dat2 = dat1.sort_values(by='viewcount', ascending=False)

    # print the top 3 viewcounts the answered question
    print('\nTop 3 viewcounts: \n', dat2[['title','viewcount']].head(3))

    # select the users (quest_name) column
    dat3 = data['quest_name']

    # remove NaN value from data
    dat3 = dat3.dropna(axis=0, how='any')

    # sort data by users frequency from greatest to smallest
    dat4 = Counter(dat3)

    # print the top 5 users
    print('\nTop 5 users: \n', dat4.most_common(5))

    # define a function for apply()
    # this function will split the tags column, the tags will be separted by ","
    def fun1(row):
        s = row.split(',')
        return s[0]

    # create a new column 'first_tag' to store the very first tag in the 'tags' column
    data['first_tag'] = data['tags'].apply(fun1)

    # print data with the new column 'first_tag'
    print('\nThis is the data with new column: \n', data)

    # show the elements in first_tag
    print("\n", data.drop_duplicates(['first_tag']), "\n")

    # ************* I'm not sure about the question, so I have different version for this step *******************

    # Version 1
    # Following are the mean, min, and max viewcount values for 'first_tag'
    dat5 = data[(data.first_tag == 'python') | (data.first_tag == 'pandas') | (data.first_tag == 'dataframe')]
    grouped = dat5.groupby('first_tag')
    print("\nThis is version 1, mean, min, and max viewcount values: \n", grouped['viewcount'].agg([np.mean, np.min, np.max]))

    # Version 2
    # filter the data which the 'tags' column contain 'python'
    dat_py = data[data.tags.str.contains('python')]
    # filter the data which the 'tags' column contain 'pandas'
    dat_pd = data[data.tags.str.contains('pandas')]
    # filter the data which the 'tags' column contain 'dataframe'
    dat_df = data[data.tags.str.contains('dataframe')]

    # print the mean, min, and max viewcount values for 'tags' which contain 'python'
    grouped_py = dat_py.groupby('tags')
    py_grouped = grouped_py['viewcount'].agg([np.mean, np.min, np.max])
    print("\nThis is version 2, mean, min, and max viewcount values for 'python': \n", py_grouped)

    # print the mean, min, and max viewcount values for 'tags' which contain 'pandas'
    grouped_pd = dat_pd.groupby('tags')
    pd_grouped = grouped_pd['viewcount'].agg([np.mean, np.min, np.max])
    print("\nThis is version 2, mean, min, and max viewcount values for 'pandas': \n", pd_grouped)

    # print the mean, min, and max viewcount values for 'tags' which contain 'dataframe'
    grouped_df = dat_df.groupby('tags')
    df_grouped = grouped_df['viewcount'].agg([np.mean, np.min, np.max])
    print("\nThis is version 2, mean, min, and max viewcount values for 'dataframe': \n", df_grouped)

    # Version 3
    # change the tags values which contain 'python' to 'python'
    dat_pyy = dat_py.copy()
    dat_pyy.loc[dat_pyy['tags'] != 'python', 'tag'] = 'python'

    # print the mean, min, and max viewcount values for 'tags' which contain 'python'
    grouped_pyy = dat_pyy.groupby('tag')
    pyy_grouped = grouped_pyy['viewcount'].agg([np.mean, np.min, np.max])
    print("\nThis is version 3, mean, min, and max viewcount values for 'python': \n", pyy_grouped)

    # change the tags values which contain 'pandas' to 'pandas'
    dat_pdd = dat_pd.copy()
    dat_pdd.loc[dat_pdd['tags'] != 'pandas', 'tag'] = 'pandas'

    # print the mean, min, and max viewcount values for 'tags' which contain 'pandas'
    grouped_pdd = dat_pdd.groupby('tag')
    pdd_grouped = grouped_pdd['viewcount'].agg([np.mean, np.min, np.max])
    print("\nThis is version 3, mean, min, and max viewcount values for 'pandas': \n", pdd_grouped)

    # change the tags values which contain 'dataframe' to 'dataframe'
    dat_dff = dat_df.copy()
    dat_dff.loc[dat_dff['tags'] != 'dataframe', 'tag'] = 'dataframe'

    # print the mean, min, and max viewcount values for 'tags' which contain 'dataframe'
    grouped_dff = dat_dff.groupby('tag')
    dff_grouped = grouped_dff['viewcount'].agg([np.mean, np.min, np.max])
    print("\nThis is version 2, mean, min, and max viewcount values for 'dataframe': \n", dff_grouped)

    # create a cross tab for all 'first_tag'
    a = pd.crosstab(index=data.answercount, columns=data.first_tag)
    print("\nFollowing is the cross tab for all 'first_tag: ", a)

    # create a cross tab only for 'python'
    data_py = data[data.first_tag == 'python']
    b = pd.crosstab(index=data_py.answercount, columns=data_py.first_tag)
    print("\nFollowing is the cross tab for 'python: ", b)


# Q3
#
def analyze_corpus(filepath):
     data = pd.DataFrame(pd.read_csv(filepath, header=0))

     title = data['title']
     title = title.str.lower()

     d = title[:20]
     arr = d.str.split(" ")
     print(arr)

     a = arr.tolist()
     print(a)

     b = list(chain(*a))
     print(b)

     c = list(set(b))
     c.sort(key=b.index)
     print(c)

     e = Counter(b)
     f = e.most_common(5)
     print(f)


# best practice to test your class
# if your script is exported as a module,
# the following part is ignored
# this is equivalent to main() in Java

if __name__ == "__main__":
    # Test Question 1
    arr = np.array([[0, 1, 0, 2, 0, 1], [1, 0, 1, 1, 2, 0], [0, 0, 2, 0, 0, 1]])

    print("\nFollowing is Q1: ")
    tf_idf, top_k = analyze_tf_idf(arr, 3)
    print('\n', top_k)


    #
    print("\nFollowing is Q2\n")
    print(analyze_data("/Users/haodong/Desktop/question.csv"))
    #
    # test question 3
    print("\nQ3")
    analyze_corpus('/Users/haodong/Desktop/question.csv')