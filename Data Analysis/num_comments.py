import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from Utility.plot_utility import plot_mean_bar_graph


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


def filter_columns(df):
    # Filter out unnecessary columns
    columns = [
        'subreddit',
        'score',
        'num_comments'
    ]
    df.drop(columns=df.columns.difference(columns), inplace=True)


def filter_low_num_comments(df):
    # Filter out num_comments with no words
    mask = df['num_comments'] >= 1
    df.drop(df[~mask].index, inplace=True)


def separate_scores_by_num_comments(df):
    median_num_comments = df['num_comments'].median()

    high_num_comments_score = df[df['num_comments'] > median_num_comments]['score']
    low_num_comments_score = df[df['num_comments'] <= median_num_comments]['score']
    
    return {
        'high_num_comments_score': high_num_comments_score,
        'low_num_comments_score':low_num_comments_score,
    }
    
    
def test_similar_distribution(high_num_comments_score, low_num_comments_score):
    plt.hist(high_num_comments_score, bins=10, alpha=0.5, label='high_num_comments_score')
    plt.hist(low_num_comments_score, bins=10, alpha=0.5, label='low_num_comments_score')

    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of high and low num_comments scores')
    plt.legend(loc='upper right')

    plt.show()


# Utility function for perform_mann_whitney_u
def interpret_mannwhitneyu(p_value, alpha=0.05):
    if p_value < alpha:
        return "The distributions of the two groups are significantly different:\n the number of comments may have an impact on the score of a post"
    else:
        return "The distributions of the two groups are not significantly different:\n the number of comments does not have a significant impact on the score of a post"


def perform_mann_whitney_u(high_num_comments_score, low_num_comments_score):
    statistic, p_value = stats.mannwhitneyu(high_num_comments_score, low_num_comments_score)

    print(f'Mann-Whitney U test statistic: {statistic}, p-value: {p_value}')
    print(interpret_mannwhitneyu(p_value))


def main():
    
    # 1. Read in the reddit submission data
    df = read_data()
    # 2. Filter out unncessary columns
    filter_columns(df)
    
    # 3. Filter out num_comments with no words
    filter_low_num_comments(df)
    
    # 4. Separate scores by num_comments
    separated_scores = separate_scores_by_num_comments(df)
    
    # 5. Test if the distributions of the two groups are similar
    high_num_comments_score, low_num_comments_score = separated_scores['high_num_comments_score'], separated_scores['low_num_comments_score']
    test_similar_distribution(high_num_comments_score, low_num_comments_score)
    
    # 6. Perform Mann-Whitney U test
    perform_mann_whitney_u(high_num_comments_score, low_num_comments_score)
    
    # 7. Plot bar graphs of mean number of comments to demonstrate signicant difference
    # Plot mean scores of high/low num_comments
    plot_mean_bar_graph(high_num_comments_score,
                        low_num_comments_score, 
                        'Mean scores of Reddit posts of high/low num_comments groups', 
                        ['High Num_comment Scores', 'Low Num_comment Scores'], 
                        'Reddit Post Scores', 
                        '../Graphs/num_comments.png')


if __name__ == '__main__':
    main()
