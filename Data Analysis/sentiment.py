import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import copy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn


def read_data():
    file_paths = [
        'one/part-00000-9076dc6d-fa59-4c36-a0cf-8808e309da7b-c000.json.gz',
        'two/part-00000-c7cf2076-eae1-4d0c-bce1-e7b0c43a3bf1-c000.json.gz',
        'three/part-00000-662e59e6-5ee7-48da-a85f-1bcf09724f97-c000.json.gz',
        'four/part-00000-3060ac52-6be0-4d42-a322-3e4a7954a4f4-c000.json.gz',
        'five/part-00000-9c42996a-80d4-4a96-b59b-228f5e241a65-c000.json.gz',
        'six/part-00000-5da94b81-55c0-42e7-b3dc-5a51a14e8589-c000.json.gz',
        'seven/part-00000-eb573e13-d85e-400c-b3db-ffa9bd2d5543-c000.json.gz',
        'eight/part-00000-61526e86-ba2d-4df5-a9d1-d043ca875b62-c000.json.gz',
        'nine/part-00000-dc7c0356-fae4-47b8-a93a-f9b401cf70f0-c000.json.gz',
        'ten/part-00000-f3ae3925-50c4-469e-8304-6007e9b4cdab-c000.json.gz',
        'eleven/part-00000-2aa1781a-723e-49c4-a488-47ca50409657-c000.json.gz',
        'twelve/part-00000-7ff282ff-fa5e-494c-ab5c-59f07d2a2f0d-c000.json.gz',
    ]

    data_frames = [pd.read_json(os.path.join('..', 'Cleaned Data', fp), lines=True) for fp in file_paths]

    return pd.concat(data_frames, ignore_index=True)


def get_cols(df):
    # get the columns we need from the dataset
    columns = [
        'title',
        'score',
        'selftext'
    ]

    df = df[columns]
    return df


def calculate_sentiment(df, analyzer):
    # deepcopy the dataframe to avoid errors
    df2 = copy.deepcopy(df)

    # get the sentiment scores of each title and selftext
    df2['title_sentiment_scores'] = df['title'].apply(analyzer.polarity_scores)
    df2['selftext_sentiment_scores'] = df['selftext'].apply(analyzer.polarity_scores)

    df = copy.deepcopy(df2)

    return df


def final_sentiment(sentiment):
    # this function returns either 'neutral', 'positive', or 'negative' depending on compound sentiment score
    compound = sentiment['compound']

    if compound >= 0.05:
        return "positive"
    elif (compound > -0.05) and (compound < 0.05):
        return "neutral"
    elif compound <= -0.05:
        return "negative"


def compound_extractor(sentiment):
    # this function extracts the compound score from the sentiment analysis
    compound = sentiment['compound']

    return compound

def get_category_sentiment(df):
    # extract the compound sentiment score from each title and selftext submission
    df['compound_title'] = df['title_sentiment_scores'].apply(compound_extractor)
    df['compound_text'] = df['selftext_sentiment_scores'].apply(compound_extractor)

    # get the positive, negative, or neutral category for each submission
    df['sentiment_final_title'] = df['title_sentiment_scores'].apply(final_sentiment)
    df['sentiment_final_selftext'] = df['selftext_sentiment_scores'].apply(final_sentiment)

    return df


def calculate_chi(count_ph, count_pl, count_nh, count_nl, count_nuh, count_nul):
    # calculate the chi2_contingency
    table = np.array([[count_pl, count_nl, count_nul], [count_ph, count_nh, count_nuh]])
    res = stats.chi2_contingency(table)
    return res


def plot_results(count_ph, count_pl, count_nh, count_nl, count_nuh, count_nul):
    # plot the chi results
    x = np.array(["High scores", "Low Scores"])
    y = np.array([[count_ph, count_pl], [count_nuh, count_nul], [count_nh, count_nl]])

    X = np.arange(2)
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1])
    plt.grid(axis='x')
    ax.bar(X + 0.00, y[0], color='skyblue', width=0.25)
    ax.bar(X + 0.25, y[1], color='palegreen', width=0.25)
    ax.bar(X + 0.50, y[2], color='lightcoral', width=0.25)

    ax.set_title('Sentiment on High vs Low Scores')
    ax.set_ylabel('Number of Scores in Sentiment Range')
    ax.set_xlabel('Reddit Scores')

    ax.set_xticks(X + 0.25)
    ax.set_xticklabels(['High Scores', 'Low Scores'])

    ax.legend(labels=['Positive', 'Neutral', 'Negative'])

    # plot the chi table
    columns = ['Positive', 'Neutral', 'Negative']
    rows = ['High Score', 'Low Score']

    data = np.array([[count_ph, count_nuh, count_nh], [count_pl, count_nul, count_nl]])

    plt.table(cellText=data, rowLabels=rows, colLabels=columns, loc='bottom', bbox=[0.14, -0.4, 0.8, 0.25])
    fig.savefig('../Graphs/sentiment_scores.png', bbox_inches='tight', pad_inches=0.1)

def main():
    # set seaborn for better graphs
    seaborn.set()

    print("program is loading and calculating. This may take several minutes please wait. . .")

    # read in data
    df = read_data()

    # get the columns we need and remove the rest
    df = get_cols(df)

    # create the sentiment analysis tool
    analyzer = SentimentIntensityAnalyzer()

    # get the sentiment scores for title and selftext
    df = calculate_sentiment(df, analyzer)

    # get the sentiment category result
    df = get_category_sentiment(df)

    # get the mean of all scores
    mean = df['score'].mean()

    # separate the submissions by sentiment, positive, negative, neutral
    positive_posts = df[df['sentiment_final_selftext'] == 'positive']
    negative_posts = df[df['sentiment_final_selftext'] == 'negative']
    neutral_posts = df[df['sentiment_final_selftext'] == 'neutral']

    # separate submissions by high or low score for each category
    pos_post_high = positive_posts[positive_posts['score'] >= mean]
    pos_post_low = positive_posts[positive_posts['score'] < mean]

    neg_score_high = negative_posts[negative_posts['score'] >= mean]
    neg_score_low = negative_posts[negative_posts['score'] < mean]

    neu_score_high = neutral_posts[neutral_posts['score'] >= mean]
    neu_score_low = neutral_posts[neutral_posts['score'] < mean]

    # count the total for each category
    count_ph = pos_post_high['score'].count()
    count_pl = pos_post_low['score'].count()
    count_nh = neg_score_high['score'].count()
    count_nl = neg_score_low['score'].count()
    count_nuh = neu_score_high['score'].count()
    count_nul = neu_score_low['score'].count()

    chi_result = calculate_chi(count_ph, count_pl, count_nh, count_nl, count_nuh, count_nul)

    print("Result of Chi:")
    print(chi_result)
    print("p-value: ", chi_result.pvalue)

    plot_results(count_ph, count_pl, count_nh, count_nl, count_nuh, count_nul)
    print("Graph saved to folder.")
    print("Program complete.")

if __name__ == '__main__':
    main()
