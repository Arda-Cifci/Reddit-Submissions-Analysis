import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Utility.plot_utility import plot_mean_bar_graph
from Utility.plot_utility_anova import plot_mean_bar_graph_3candidates

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
    ]
    df.drop(columns=df.columns.difference(columns), inplace=True)


def filter_nan_subreddit(df):
    # Filter out rows with no subreddit name
    mask = df['subreddit'].isna()
    df.drop(df[mask].index, inplace=True)


def groupby_subreddit_size(df):
    subreddit_popularity = df.groupby('subreddit').size()
    subreddit_popularity = subreddit_popularity.sort_values()
    
    df['subreddit_popularity'] = df['subreddit'].map(subreddit_popularity)


def separate_scores_by_subreddit_popularity(df):
    median_subreddit_popularity = df['subreddit_popularity'].median()

    high_subreddit_popularity_score = df[df['subreddit_popularity'] > median_subreddit_popularity]['score']
    low_subreddit_popularity_score = df[df['subreddit_popularity'] <= median_subreddit_popularity]['score']
        
    return {
        'high_subreddit_popularity_score': high_subreddit_popularity_score,
        'low_subreddit_popularity_score':low_subreddit_popularity_score,
    }


def test_similar_distribution(high_subreddit_popularity_score, low_subreddit_popularity_score):
    plt.hist(high_subreddit_popularity_score, bins=10, alpha=0.5, label='high_subreddit_popularity_score')
    plt.hist(low_subreddit_popularity_score, bins=10, alpha=0.5, label='low_subreddit_popularity_score')

    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Scores for High and Low Subreddit_Popularity')
    plt.legend(loc='upper right')


def interpret_mannwhitneyu(p_value, alpha=0.05):
    # Utility function for perform_mann_whitney_u
    if p_value < alpha:
        return "The distributions of the two groups are significantly different:\n the popularity of subreddits may have an impact on the score of a post"
    else:
        return "The distributions of the two groups are not significantly different:\n the popularity of subreddits does not have a significant impact on the score of a post"
    
    
def perform_mann_whitney_u(high_subreddit_popularity_score, low_subreddit_popularity_score):
    statistic, p_value = stats.mannwhitneyu(high_subreddit_popularity_score, low_subreddit_popularity_score)

    print(f'Mann-Whitney U test statistic: {statistic}, p-value: {p_value}')
    print(interpret_mannwhitneyu(p_value))


def perform_normal_test(df):
    statistic, p_value = stats.normaltest(df['subreddit_popularity'])

    print(f'p_value: {p_value}')

    # Print the normality test result
    if p_value < 0.05:
        print("The column subreddit_popularity is not normally distributed (p-value < 0.05).")
    else:
        print("The column subreddit_popularity is approximately normally distributed (p-value >= 0.05).")


def transform_subreddit_popularity(df):
    '''
    .exp results in inf subreddit_popularity
    .square results in left-skewed histogram
    .log results in somewhat normal histogram
    .sqrt results in semi left-skewed histogram
    '''
    #df['subreddit_popularity_exp'] = np.exp(df['subreddit_popularity']) 
    df['subreddit_popularity_square'] = np.square(df['subreddit_popularity'])
    df['subreddit_popularity_log'] = np.log(df['subreddit_popularity'] + 1) 
    df['subreddit_popularity_sqrt'] = np.sqrt(df['subreddit_popularity'])


def separate_scores_by_low_medium_high(df):
    df['subreddit_category'] = pd.qcut(df['subreddit_popularity_log'], 3, labels=['low', 'medium', 'high'])

    low_popularity_anova = df[df['subreddit_category'] == 'low']['score']
    medium_popularity_anova = df[df['subreddit_category'] == 'medium']['score']
    high_popularity_anova = df[df['subreddit_category'] == 'high']['score']
    
    return {
        'low_popularity_anova': low_popularity_anova,
        'medium_popularity_anova':medium_popularity_anova,
        'high_popularity_anova':high_popularity_anova,
    }


def interpret_anova(p_value, alpha=0.05):
    # Utility function for perform_anova
    if p_value < alpha:
        return "The distributions of the three groups are significantly different:\n the popularity of subreddits may have an impact on the score of a post"
    else:
        return "The distributions of the three groups are not significantly different:\n the popularity of subreddits does not have a significant impact on the score of a post"
    
    
def perform_anova(low_popularity, medium_popularity, high_popularity):
    statistic, p_value = stats.f_oneway(low_popularity, medium_popularity, high_popularity)

    print(f'ANOVA one-way test statistic: {statistic}, p-value: {p_value}')
    print(interpret_anova(p_value))
    

def main():
    
    # 1. Read in the reddit submission data
    df = read_data()

    # 2. Filter out unncessary columns
    filter_columns(df)
    
    # 3. Filter out NaN subreddits
    filter_nan_subreddit(df)
    
    # 4. Group by subreddit size
    groupby_subreddit_size(df)
    
    # 5. Separate scores by subreddit popularity
    separated_scores = separate_scores_by_subreddit_popularity(df)
    high_subreddit_popularity_score, low_subreddit_popularity_score = separated_scores['high_subreddit_popularity_score'], separated_scores['low_subreddit_popularity_score']
    
    # 6. Test if the distributions of the two groups are similar
    test_similar_distribution(high_subreddit_popularity_score, low_subreddit_popularity_score)
    
    # 6. Perform Mann-Whitney U test
    perform_mann_whitney_u(high_subreddit_popularity_score, low_subreddit_popularity_score)
    
    # 7. Plot bar graphs of mean number of comments to demonstrate signicant difference
    plot_mean_bar_graph(high_subreddit_popularity_score,
                        low_subreddit_popularity_score, 
                        'Mean scores of Reddit posts of high/low subreddit_popularity groups', 
                        ['High Subreddit_Popularity Scores', 'Low Subreddit_Popularity Scores'], 
                        'Reddit Post Scores', 
                        '../Graphs/subreddit_popularity.png')

    # 8. Perform normal test on subreddit_popularity
    perform_normal_test(df)
    
    # 9. Transform subreddit_popularity to try to make it more normal
    transform_subreddit_popularity(df)
    
    # 10. Plot histogram of transformed subreddit_popularity
    plt.hist(df['subreddit_popularity_log'], bins=50)
    plt.xlabel('Post Subreddit Popularity')
    plt.ylabel('Frequency')
    plt.title('Histogram of log transformed subreddit_popularity')
    plt.show()
    #plt.savefig('../Graphs/subreddit_popularity_log_transformed_histogram.png')
    
    '''
    subreddit_popularity_log appears normal enough, so perform ANOVA
    '''
    
    # 11. Separate scores by low/medium/high subreddit_popularity
    separated_scores_anova = separate_scores_by_low_medium_high(df)
    low_popularity_anova, medium_popularity_anova, high_popularity_anova = separated_scores_anova['low_popularity_anova'], separated_scores_anova['medium_popularity_anova'], separated_scores_anova['high_popularity_anova'] 

    # 12. Perform ANOVA
    perform_anova(low_popularity_anova, medium_popularity_anova, medium_popularity_anova)
    
    # 13. Plot bar graphs of mean number of comments to demonstrate signicant difference
    plot_mean_bar_graph_3candidates(medium_popularity_anova,
                    medium_popularity_anova, 
                    low_popularity_anova,
                    'Mean scores of Reddit posts of high/medium/low subreddit_popularity groups', 
                    ['High Popularity', 'Medium Popularity', 'Low Popularity'], 
                    'Reddit Post Scores', 
                    '../Graphs/subreddit_popularity_anova.png')
    
    
if __name__ == '__main__':
    main()