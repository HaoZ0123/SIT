# ALITA: BATTLE ANGEL
# https://www.rottentomatoes.com/m/alita_battle_angel/reviews/
# Haodong Zhao    10409845

import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Q1
# define the getData() function
def getData(movie_id):
    print('Following is Q1: ', '\n')
    page = requests.get('https://www.rottentomatoes.com/m/{}/reviews/'.format(movie_id))

    # create a list for output
    data = []

    soup = BeautifulSoup(page.content, 'html.parser')
    divs = soup.select('div#reviews div div div.review_table_row')

    # scrape info from page
    for idx, div in enumerate(divs):
        idx += 1

    # initial output values
        date = None
        description = None
        score = None

    # find the date of reviews
        d_date = div.select('div.review_date')
        date = d_date[0].get_text()
        # print(date)

    # find the description of reviews
        d_description = div.select('div.the_review')
        description = d_description[0].get_text()
        # print(description)

    # find the score of reviews
        d_score = div.select('div.small.subtle')
        score = d_score[1].get_text().split(':')

    # display numerical rating scores and display 'None' for alphabetic reviews and no rating reviews
        if len(score) > 1:
            if 3 < len(score[1]) < 7:
                score = score[1].strip()
            else:
                score = 'None'
        else:
            score = 'None'

    # put all the three output in list
        data.append((date, description, score))

    # for this question, we only need 20 outputs
        if idx == 20:
            break

    return data


# Q2
# define a function for data ploting
def plot_data(data):

    # preparing for convert rating to decimals
    rate1 = []
    rate2 = []

    # preparing for plot
    year = []
    rating = []

    # set numerical rating scores and set 'NaN' for alphabetic reviews and no rating reviews
    for i in range(0, len(data)):
        rate = data[i][-1].split('/')
        if len(rate) > 1:
            rate1.append(rate[0])
            rate2.append(rate[1])
        else:
            rate1.append(np.nan)
            rate2.append(np.nan)

    # find year
    for i in range(0, len(data)):
        y = data[i][0]
        year.append(y[-4:])

    # convert rating scores to decimals
    for i in range(0, len(rate1)):
        try:
            rating.append(float(rate1[i]) / float(rate2[i]))
        except:
            rating.append('None')

    # create a dataframe for year and rating
    c = {'year': year, 'rating': rating}
    df = pd.DataFrame(c)

    # drop NA values for mean value calculation
    a = df.dropna(axis=0, how='any')

    # group the dataframe by 'year' attr
    grouped = a.groupby('year')

    # calculate the average rating by year
    df1 = grouped.mean()

    # create the bar chart and display
    data = df1.plot(kind = 'bar', title = 'Average Rating by Year')
    plt.show()
    return data


# Q3
# define function for scrape reviews in all pages
def getFullData(movie_id):
    print('\n')

    # find the original (1st) page
    page = requests.get('https://www.rottentomatoes.com/m/{}/reviews/'.format(movie_id))

    # create a list for output
    data = []

    soup = BeautifulSoup(page.content, 'html.parser')

    # find the number of pages
    num = soup.select('div#reviews div div span.pageInfo')
    n = int(num[0].get_text().split()[-1])

    # for loop to scrape all pages by using the number of pages we got
    for i in range(1, n + 1):

        # find page by using different url each time
        page = requests.get('https://www.rottentomatoes.com/m/{}/reviews/?page={}&sort='.format('finding_dory', i))
        soup = BeautifulSoup(page.content, 'html.parser')
        divs = soup.select('div#reviews div div div.review_table_row')

        # scrape status each time
        print('Scraping page', i, page)

        # following block is same as Q1
        for idx, div in enumerate(divs):
            idx += 1

            date = None
            description = None
            score = None

            d_date = div.select('div.review_date')
            date = d_date[0].get_text().strip()

            d_description = div.select('div.the_review')
            description = d_description[0].get_text().strip()

            d_score = div.select('div.small.subtle')
            score = d_score[1].get_text().split(':')

            if len(score) > 1:
                if 3 < len(score[1]) < 8:
                    score = score[1].strip()
                else:
                    score = 'None'
            else:
                score = 'None'

            data.append((date, description, score))

        # constraints for this for loop
        i += 1
        if i > n:
            break
    return data

    # following block is same as Q2
    rate1 = []
    rate2 = []
    year = []
    rating = []
    for i in range(0, len(data)):
        rate = data[i][-1].split('/')
        if len(rate) > 1:
            rate1.append(rate[0])
            rate2.append(rate[1])
        else:
            rate1.append(np.nan)
            rate2.append(np.nan)

    for i in range(0, len(data)):
        y = data[i][0]
        year.append(y[-4:])

    for i in range(0, len(rate1)):
        try:
            rating.append(float(rate1[i]) / float(rate2[i]))
        except:
            rating.append('None')

    c = {'year': year, 'rating': rating}
    df = pd.DataFrame(c)

    a = df.dropna(axis=0, how='any')

    grouped = a.groupby('year')
    df1 = grouped.mean()

    df1['rating'].plot('bar')
    plt.show()


if __name__ == "__main__":


# Test Q1
    data = getData("finding_dory")
    print(data)

# Test Q2
    plot_data(data)

# Test Q3
    print('\nFollowing is Q3:', )
    data=getFullData("finding_dory")
    print(len(data), '\n', data[-1])
    plot_data(data)