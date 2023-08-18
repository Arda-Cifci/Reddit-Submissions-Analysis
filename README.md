# Reddit Post Analysis

This project aims to uncover the underlying factors that contribute to the popularity of Reddit posts, measured through their scores. The project is divided into several parts, where each one investigates a different aspect of a Reddit post: readability, the number of comments, subreddit popularity, post length, date and time of post, and the overall sentiment of the post.

## Required Libraries

Please install the following Python libraries before running the project:

- Pandas
- Numpy
- Matplotlib
- Textstat
- Scipy
- Statsmodels
- Seaborn
- vaderSentiment.vaderSentiment

```bash
pip install pandas numpy matplotlib textstat scipy statsmodels seaborn vadersentiment
```

## Other Requirements

- PySpark Version 3.2+
- Jupyter Notebook
- Python Version 3.8+

## Running the Project

The script to gather and clean the data is:

- gather_clean.py

The main scripts in this project are:

- readability_analysis.py
- comments_analysis.py
- subreddit_popularity_analysis.py
- post_length_analysis.py
- submission_byhour.py
- sentiment.py

The gather and clean script should be run on the SFU cluster with: 
```bash
spark-submit gather_clean.py /courses/datasets/reddit_submissions_repartitioned/year=2016/month=01/*.json.gz output
```
replaceing each month with the next (month=02, month=03, etc) to obtain all 12 required cleaned data files.  You can extract the cleaned data by copying the hdfs output to local and then scp it to your personal computer if desired. 

You can run each main script independently with Python:

```bash
python readability_analysis.py
python comments_analysis.py
python subreddit_popularity_analysis.py
python post_length_analysis.py
python submission_byhour.py
python sentiment.py
```
## Files Produced

The gather and clean script produces the data files required for the project. These cleaned data files are saved in `Cleaned Data` seperated by month.

gather_clean.py
 - Should produce 1 cleaned file every time it is run.
 - Run 12 times with the above command and different months to get all 12 files of cleaned data from the cluster.

Each main script generates various plots and prints statistical analysis results to the console. These plots are saved in the `Graphs` directory.

readability_analysis.py
 - `selftext_grade_bar.png` , `selftext_readability_bar.png`, `title_grade_bar.png`, `title_readability_bar.png`

comments_analysis.py
 - `num_comments.png`

subreddit_popularity_analysis.py
 - `subreddit_popularity.png`, `subreddit_popularity_anova.png`

post_length_analysis.py
 - `post_length.png`, `post_length_anova.png`

submission_byhour.py
 - `average_submission_by_hour.png`, `residuals_submission_by_hour.png`

sentiment.py
 - `sentiment_scores.png`

## Note

The project uses data from Reddit's 2016 submissions, which have been preprocessed and cleaned for the purposes of this analysis. The dataset only includes text-based posts, and any posts with missing or irrelevant information were excluded.

There are both py and notebook versions of the main scripts, either version will produce the same results.

## Group Members

- Arda
- Ryan

## Language

- Python
