import os
import pandas as pd
from scipy import stats
import textstat.textstat
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
        'title',
        'score',
        'selftext'
    ]
    df.drop(columns=df.columns.difference(columns), inplace=True)


def filter_low_selftext(df):
    # Filter out selftext with no words
    mask = df['selftext'].apply(lambda x: len(str(x).split()) >= 1) & ~(df['selftext'].isin(['[removed]', '[deleted]']))
    df.drop(df[~mask].index, inplace=True)


def calculate_readability(df):
    # Perform readability score
    df['title_readability'] = df['title'].apply(textstat.flesch_reading_ease)
    df['selftext_readability'] = df['selftext'].apply(
        textstat.flesch_reading_ease)

    df['title_grade'] = df['title'].apply(
        textstat.dale_chall_readability_score)
    df['selftext_grade'] = df['selftext'].apply(
        textstat.dale_chall_readability_score)


def test_correlation_to_score(df):
    # -1 to 1 (positive/negative) & 0 indicates NO relationship
    correlations = {
        'selftext_readability': df['score'].corr(df['selftext_readability']),
        'title_readability': df['score'].corr(df['title_readability']),
        'selftext_grade': df['score'].corr(df['selftext_grade']),
        'title_grade': df['score'].corr(df['title_grade'])
    }

    threshold = 0.1

    for column, correlation in correlations.items():
        print(f'Correlation between score and {column}: {correlation}')
        if abs(correlation) > threshold:
            print('> 0.1: There is a significant correlation.')
        else:
            print('<= 0.1: There is no significant correlation.')


def separate_scores_by_readability(df):
    # Separte series of Reddit submission scores from submissions with high/low selftext/title readability score

    # Separte scores by low / high selftext readability
    high_selftext_readability = df[df['selftext_readability']
                                   > df['selftext_readability'].median()]['score']
    low_selftext_readability = df[df['selftext_readability']
                                  <= df['selftext_readability'].median()]['score']

    # Separte scores by low / high title readability
    high_title_readability = df[df['title_readability']
                                > df['title_readability'].median()]['score']
    low_title_readability = df[df['title_readability']
                               <= df['title_readability'].median()]['score']

    # Separte scores by low / high selftext grade
    high_selftext_grade = df[df['selftext_grade']
                             > df['selftext_grade'].median()]['score']
    low_selftext_grade = df[df['selftext_grade']
                            <= df['selftext_grade'].median()]['score']

    # Separte scores by low / high title grade
    high_title_grade = df[df['title_grade'] >
                          df['title_grade'].median()]['score']
    low_title_grade = df[df['title_grade'] <=
                         df['title_grade'].median()]['score']
    
    return {
        'high_selftext_readability': high_selftext_readability,
        'low_selftext_readability':low_selftext_readability,
        'high_title_readability':high_title_readability,
        'low_title_readability':low_title_readability,
        'high_selftext_grade':high_selftext_grade,
        'low_selftext_grade':low_selftext_grade,
        'high_title_grade':high_title_grade,
        'low_title_grade':low_title_grade,
    }


# Utility function for test_normal_distribution
def normality_category(p_value, alpha=0.05):
    if p_value < alpha:
        return "< 0.05: Not a normal distribution"
    elif p_value < 0.1:
        return "> 0.05 & < 0.1: Normal enough"
    else:
        return "> 0.1: Normal"


def test_normal_distribution(candidate_dict):
    # Perform a normal test on the separated data
    for candidate_name, candidate_series in candidate_dict.items():
        statistic, p_value = stats.normaltest(candidate_series)
        print(f'{candidate_name}_p = {p_value}:\n {normality_category(p_value)}')


# Utility function for perform_t_test
def ttest_category(p_value, alpha=0.05):
    if p_value < alpha:
        return "< 0.05: There is a significant difference between the groups"
    else:
        return ">= 0.05: There is no significant difference between the groups"


def perform_t_test(candidate_dict):
    # < 0.05 indicates a significant difference between the groups
    # -> different (high/low) readability scores make a difference in the score for Reddit submissions
    keys = list(candidate_dict.keys())
    for i in range(0, len(keys), 2):
        statistic, p_value = stats.ttest_ind(
            candidate_dict[keys[i]], candidate_dict[keys[i+1]], equal_var=False)
        print(f'{keys[i]} vs {keys[i+1]}:\n {ttest_category(p_value)}')


def main():

    # 1. Read in the reddit submission data
    df = read_data()

    # 2. Filter out unncessary columns
    filter_columns(df)
    
    # 3. Filter out selftext with no words
    filter_low_selftext(df)

    # 4. Perform readability scores
    calculate_readability(df)
    
    # 5. Test correlation between readability scores and score
    test_correlation_to_score(df)
    
    # 6. Separate scores by low/high readability scores
    separated_scores = separate_scores_by_readability(df)
    
    # 7. Perform a normal test on the separated data
    test_normal_distribution(separated_scores)
    
    # 8. Perform Ttest on the separated data
    perform_t_test(separated_scores)
    
    # 9. Plot bar graphs of mean scores to demonstrate significant difference  
    # Plot mean scores of high/low selftext readability
    plot_mean_bar_graph(separated_scores['high_selftext_readability'],
                        separated_scores['low_selftext_readability'], 
                        'Mean scores by selftext readability', 
                        ['High Selftext Readability', 'Low Selftext Readability'], 
                        'Scores', 
                        '../Graphs/selftext_readability_bar.png')
    
    plot_mean_bar_graph(separated_scores['high_title_readability'],
                        separated_scores['low_title_readability'], 
                        'Mean scores by Title readability', 
                        ['High Title Readability', 'Low Title Readability'], 
                        'Scores', 
                        '../Graphs/title_readability_bar.png')

    # Plot mean scores of high/low selftext grade
    plot_mean_bar_graph(separated_scores['high_selftext_grade'],
                        separated_scores['low_selftext_grade'], 
                        'Mean scores by selftext grade', 
                        ['High Selftext Grade', 'Low Selftext Grade'], 
                        'Scores', 
                        '../Graphs/selftext_grade_bar.png')
    
    # Plot mean scores of high/low title grade
    plot_mean_bar_graph(separated_scores['high_title_grade'],
                        separated_scores['low_title_grade'], 
                        'Mean scores by title grade', 
                        ['High Title Grade', 'Low Title Grade'], 
                        'Scores', 
                        '../Graphs/title_grade_bar.png')
    

if __name__ == '__main__':
    main()