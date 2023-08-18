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
        'selftext',
    ]
    df.drop(columns=df.columns.difference(columns), inplace=True)
    
    
def filter_low_selftext(df):
    # Filter out selftext with no words
    mask = df['selftext'].apply(lambda x: len(str(x).split()) >= 1) & ~(df['selftext'].isin(['[removed]', '[deleted]']))
    df.drop(df[~mask].index, inplace=True)
    

def calculate_post_length(df):
    df['post_length'] = df['selftext'].str.len()
    df = df.sort_values('post_length')


def separate_scores_by_post_length(df):
    median_post_length = df['post_length'].median()
    
    high_post_length_score = df[df['post_length'] > median_post_length]['score']
    low_post_length_score = df[df['post_length'] <= median_post_length]['score']
    
    return {
        'high_post_length_score': high_post_length_score,
        'low_post_length_score':low_post_length_score,
    }
    
    
def test_similar_distribution(high_post_length_score, low_post_length_score):
    plt.hist(high_post_length_score, bins=10, alpha=0.5, label='high_post_length_score')
    plt.hist(low_post_length_score, bins=10, alpha=0.5, label='low_post_length_score')

    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title('Histogram of Scores for High and Low Post_Length')
    plt.legend(loc='upper right')

    plt.show()


# Utility function for perform_mann_whitney_u
def interpret_mannwhitneyu(p_value, alpha=0.05):
    if p_value < alpha:
        return "The distributions of the two groups are significantly different:\n the length of posts may have an impact on the score of a post"
    else:
        return "The distributions of the two groups are not significantly different:\n the length of posts does not have a significant impact on the score of a post"
    
    
def perform_mann_whitney_u(high_post_length_score, low_post_length_score):
    statistic, p_value = stats.mannwhitneyu(high_post_length_score, low_post_length_score)

    print(f'Mann-Whitney U test statistic: {statistic}, p-value: {p_value}')
    print(interpret_mannwhitneyu(p_value))


def perform_normal_test(df):
    statistic, p_value = stats.normaltest(df['post_length'])

    print(f'p_value: {p_value}')

    # Print the normality test result
    if p_value < 0.05:
        print("The column post_length is not normally distributed (p-value < 0.05).")
    else:
        print("The column post_length is approximately normally distributed (p-value >= 0.05).")


def transform_post_length(df):
    '''
    .exp results in inf post_length
    .square results in left-skewed histogram
    .log results in somewhat normal histogram
    .sqrt results in semi left-skewed histogram
    '''
    #df['post_length_exp'] = np.exp(df['post_length']) 
    df['post_length_square'] = np.square(df['post_length'])
    df['post_length_log'] = np.log(df['post_length'] + 1) 
    df['post_length_sqrt'] = np.sqrt(df['post_length']) 


def separate_scores_by_low_medium_high(df):
    df['post_length_category'] = pd.qcut(df['post_length_log'], 3, labels=['low', 'medium', 'high'])

    low_post_length_anova = df[df['post_length_category'] == 'low']['score']
    medium_post_length_anova = df[df['post_length_category'] == 'medium']['score']
    high_post_length_anova = df[df['post_length_category'] == 'high']['score']
    
    return {
        'low_post_length_anova': low_post_length_anova,
        'medium_post_length_anova':medium_post_length_anova,
        'high_post_length_anova':high_post_length_anova,
    }


# Utility function for perform_anova
def interpret_anova(p_value, alpha=0.05):
    if p_value < alpha:
        return "The distributions of the three groups are significantly different:\n the length of posts may have an impact on the score of a post"
    else:
        return "The distributions of the three groups are not significantly different:\n the length of posts does not have a significant impact on the score of a post"
    
    
def perform_anova(low_post_length, medium_post_length, high_post_length):
    statistic, p_value = stats.f_oneway(low_post_length, medium_post_length, high_post_length)

    print(f'ANOVA one-way test statistic: {statistic}, p-value: {p_value}')
    print(interpret_anova(p_value))
    

def main():
    
    # 1. Read in the reddit submission data
    df = read_data()

    # 2. Filter out unncessary columns
    filter_columns(df)
    
    # 3. Filter out NaN subreddits
    filter_low_selftext(df)
    
    # 4. Calculate post length
    calculate_post_length(df)
    
    # 5. Separate scores by post length
    separated_scores = separate_scores_by_post_length(df)
    high_post_length_score, low_post_length_score = separated_scores['high_post_length_score'], separated_scores['low_post_length_score']
    
    # 6. Test if the distributions of the two groups are similar
    test_similar_distribution(high_post_length_score, low_post_length_score)
    
    # 6. Perform Mann-Whitney U test
    perform_mann_whitney_u(high_post_length_score, low_post_length_score)
    
    # 7. Plot bar graphs of mean number of comments to demonstrate signicant difference
    plot_mean_bar_graph(high_post_length_score,
                        low_post_length_score, 
                        'Mean scores of Reddit posts of high/low post_length groups', 
                        ['High Post Length', 'Low Post Length Scores'], 
                        'Reddit Post Scores', 
                        '../Graphs/post_length.png')

    # 8. Perform normal test on post_length
    perform_normal_test(df)
    
    # 9. Transform post_length to try to make it more normal
    transform_post_length(df)
    
    # 10. Plot histogram of transformed post_length
    plt.hist(df['post_length_log'], bins=50)
    plt.xlabel('Post Length')
    plt.ylabel('Frequency')
    plt.title('Histogram of log transformed post_length')
    plt.show()
    # plt.savefig('../Graphs/post_length_log_transformed_histogram.png')
    
    '''
    post_length_log appears normal enough, so perform ANOVA
    '''
    
    # 11. Separate scores by low/medium/high post_length
    separated_scores_anova = separate_scores_by_low_medium_high(df)
    low_post_length_anova, medium_post_length_anova, high_post_length_anova = separated_scores_anova['low_post_length_anova'], separated_scores_anova['medium_post_length_anova'], separated_scores_anova['high_post_length_anova'] 

    # 12. Perform ANOVA
    perform_anova(low_post_length_anova, medium_post_length_anova, high_post_length_anova)
    
    # 13. Plot bar graphs of mean number of comments to demonstrate signicant difference
    plot_mean_bar_graph_3candidates(high_post_length_anova,
                        medium_post_length_anova, 
                        low_post_length_anova,
                        'Mean scores of Reddit posts of high/medium/low post_length groups', 
                        ['High Post Length', 'Medium Post Length', 'Low Post Length'], 
                        'Reddit Post Scores', 
                        '../Graphs/post_length_anova.png')
        
        
if __name__ == '__main__':
    main()