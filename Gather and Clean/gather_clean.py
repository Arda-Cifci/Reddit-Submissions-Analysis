from pyspark.sql import SparkSession, functions, types
import sys

assert sys.version_info >= (3, 8)  # make sure we have Python 3.8+

spark = SparkSession.builder.appName('Gather and Clean Reddit').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 8)  # make sure we have Python 3.8+
assert spark.version >= '3.2'  # make sure we have Spark 3.2+


def filter_unwanted_data(df):
    # if there is a null or none in an important part of the data remove the row
    # additionally check if title or selftext is empty or has just spaces, etc and remove them

    filtered_data = df.filter(
        (df['score'].isNotNull()) &
        (df['num_comments'].isNotNull()) &
        (df['ups'].isNotNull()) &
        (df['created_utc'].isNotNull()) &
        (df['subreddit'].isNotNull()) &
        (df['author'].isNotNull()) &
        (df['title'].isNotNull()) &
        (df['selftext'].isNotNull()) &
        (df['subreddit_id'].isNotNull()) &
        # Added the following to refine filter - Ryan 2023 July 26
        (df['over_18'] == False) &
        (df['is_self'] == True) &
        # Added the following to refine the dataset - Arda Cifci - 2023 July 31
        (df['selftext'] != '[removed]') & (df['selftext'] != '[deleted]') &
        (df['title'] != '[removed]') & (df['title'] != '[deleted]') &
        (df['word_count_self'] >= 1) &
        (df['word_count_title'] >= 1) &
        (df['selftext'] != ' ') &
        (df['selftext'] != "  ") &
        (df['selftext'] != "   ") &
        (df['selftext'] != '.') &
        (df['selftext'] != '') &
        (df['title'] != '') &
        (df['title'] != ' ') &
        (df['title'] != "  ") &
        (df['title'] != "   ") &
        (df['title'] != '.')
    )

    return filtered_data


def select_columns(df):
    # select the final columns we want
    df = df.select(
        df['name'],
        df['downs'],
        # df['permalink'],
        df['ups'],
        # df['domain'],
        df['hide_score'],
        # df['secure_media_embed'],
        # df['retrieved_on'],
        df['subreddit'],
        df['link_flair_css_class'],
        df['locked'],
        df['num_comments'],
        # df['saved'],
        # df['quarantine'],
        # df['edited'],
        df['id'],
        df['preview'],
        df['link_flair_text'],
        df['score'],
        df['author'],
        df['author_flair_css_class'],
        df['stickied'],
        df['title'],
        df['selftext'],
        # df["created_utc"],
        df['over_18'],
        df['author_flair_text'],
        df['thumbnail'],
        df['gilded'],
        # df['distinguished'],
        # df['contest_mode'],
        df['subreddit_id'],
        # df['media_embed'],
        # df['archived'],
        df['is_self'],
        # df['datetime_pst'],
        # df['time_pst'],
        df['date'],
        df['datetime'],
        # df['secure_media'],
        # df['post_hint'],
        # df['url'],
        # df['media'],
        # df['mobile_ad_url'],
        # df['disable_comments'],
        # df['promoted'],
        # df['imp_pixel'],
        # df['adserver_click_url'],
        # df['promoted_by'],
        # df['adserver_imp_pixel'],
        # df['third_party_tracking'],
        # df['href_url'],
        # df['promoted_display_name'],
        # df['original_link'],
        # df['promoted_url'],
        # df['third_party_tracking_2']
        df['word_count_self'],
        df['word_count_title']
    )

    return df


def fix_date(df):
    # make the date column into something human-readable, specifically timestamp type.
    df = df.withColumn("datetime", df.created_utc.cast(types.TimestampType()))
    df = df.withColumn('date', df['datetime'].cast('date'))
    return df


def one_word(df):
    # trim the front and back of each submission's title and selftext
    df = df.withColumn("trim_self", functions.trim(df["selftext"]))
    df = df.withColumn("trim_title", functions.trim(df["title"]))

    # add the word counts of each submission's title and selftext to a new column
    df = df.withColumn('word_count_self', functions.size(functions.split(functions.col('trim_self'), ' ')))
    df = df.withColumn('word_count_title', functions.size(functions.split(functions.col('trim_title'), ' ')))

    return df


def main(in_directory, out_directory):
    # put input file into dataframe
    reddit_data = spark.read.json(in_directory)

    # randomize the rows
    reddit_data = reddit_data.orderBy(functions.rand())

    # get a sample subset of the data, with default random seed
    reddit_data = reddit_data.sample(0.3)

    # get the number of words in body text and title
    reddit_data = one_word(reddit_data)

    # remove rows with missing important data
    reddit_data = filter_unwanted_data(reddit_data)

    # change date from epoch utc into spark Timestamp Type
    reddit_data = fix_date(reddit_data)

    # drop old created_utc date, its unneeded
    reddit_data = reddit_data.drop(reddit_data.created_utc)

    # select columns we want to keep / remove columns we have no use for
    cleaned_data = select_columns(reddit_data)

    # limit the sample to 25,000 rows (25,000 rows for each month)
    cleaned_data = cleaned_data.limit(25000)

    # output as json gz
    cleaned_data.write.json(
        out_directory, compression='gzip', mode='overwrite')


if __name__ == '__main__':
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)
