import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn
import os

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



def get_averages(data):
    # group the data by their hour
    grouped = data.groupby(data['datetime'].dt.hour)

    # get the average score for each hour
    averages_per_hour = grouped['score'].mean()

    return averages_per_hour


def create_fit(averages):
    # create a linear regression: x-axis = each hour, y-axis = the average scores for each hour
    fit_created = stats.linregress(range(24), averages)

    return fit_created


def plot_results(averages_per_hour, fit):
    # plot the result and best fit line
    plt.plot(range(24), averages_per_hour, 'b.', alpha=1, markersize=15)
    plt.plot(range(24), averages_per_hour, 'b-', alpha=0.6, linewidth=3)
    plt.plot(range(24), range(24) * fit.slope + fit.intercept, '-', linewidth=3, c='lightcoral')

    plt.title("Average Submission Score in Each Hour")
    plt.xlabel("Hours (24) - PST")
    plt.ylabel("Average Scores")

    plt.xlim(xmin=0.0, xmax=23)
    plt.ylim(ymin=0.0)

    plt.xticks(range(24))
    plt.grid(axis='x')
    plt.fill_between(range(24), averages_per_hour, alpha=0.2)
    plt.tight_layout()

    plt.savefig('../Graphs/average_submission_by_hour.png')


def fix_date(df):
    # convert the spark timestamp type into datetime
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_convert('Etc/GMT+8')

    return df


def plot_residuals(averages, fit):
    # get and plot the residuals - residuals look normal enough for me
    plt.close()
    residuals = averages - (range(24) * fit.slope + fit.intercept)
    plt.grid(axis='x')
    plt.hist(residuals)
    plt.savefig('../Graphs/residuals_submission_by_hour.png')


def main():
    # set seaborn for better graphs
    seaborn.set()

    print("program is loading and calculating, please wait a few moments. . .")

    # read in data
    data = read_data()

    # fix date - convert the spark timestamp type into datetime
    data = fix_date(data)

    # get averages for each hour
    averages = get_averages(data)

    # create a linear fit for the averages
    fit = create_fit(averages)

    # plot the results and best fit line
    plot_results(averages, fit)

    print("Plots have been saved into folder.")

    # print out the useful values
    print("p-value:", fit.pvalue)
    print("r-value:", fit.rvalue)
    print("r-value squared:", fit.rvalue**2)

    # plot the residuals
    plot_residuals(averages, fit)


if __name__ == '__main__':
    main()
