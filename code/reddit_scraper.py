import praw
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

def reddit_scraper(client_id, client_secret, user_agent, num_posts, subreddit_name, interval, time_filter, top_comments_count, output_file):
    
    """
    Class for scraping Reddit posts and comments using the praw library.

    """
    class RedditScraper:
        def __init__(self, client_id, client_secret, user_agent):
            """
            Initialize the RedditScraper with client_id, client_secret, and user_agent.

            Args:
                client_id: Reddit application client id
                client_secret: Reddit application client secret
                user_agent: User agent for Reddit application

            Returns:
                None
            """
            # Create a Reddit instance using praw
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent)

        
        def fetch_posts(self, num_posts, sub_name, interval):
            """
            Fetch posts from a specific subreddit.

            Args:
                num_posts: Number of posts to fetch
                sub_name: Name of the subreddit to fetch posts from
                interval: Time interval to fetch posts from

            Returns:
                data: List of dictionaries containing post data
                posts_list: List of praw.models.reddit.submission.Submission objects
            """
            # Get the subreddit
            subreddit = self.reddit.subreddit(sub_name)
            # Fetch the top posts based on the time filter
            posts = subreddit.top(time_filter=str(time_filter), limit=num_posts)
            # Sort the posts by creation time
            posts_list = list(posts)
            posts_list.sort(key=lambda post: post.created_utc, reverse=True)

            # Define the intervals for fetching posts
            intervals = {
                'daily': timedelta(days=1),
                'weekly': timedelta(weeks=1),
                'monthly': timedelta(weeks=4)}

            # Initialize variables for storing posts and intervals
            end_time = datetime.utcfromtimestamp(posts_list[0].created_utc)
            nested_posts = []
            current_interval_start = end_time
            data = []
            interval_num = 0

            # Loop through the posts and store them in the data list
            for post in posts_list:
                post_time = datetime.utcfromtimestamp(post.created_utc)

                if post_time < current_interval_start - intervals[interval]:
                    interval_num += 1
                    current_interval_start = post_time

                data.append({
                    'Post/Comment': 'Post',
                    'ID': post.id,
                    'Text': post.title + post.selftext,
                    'Creation Date': datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
                    'Interval Number': interval_num})

            # Count the number of posts in each interval
            interval_counts = Counter([entry['Interval Number'] for entry in data])

            return data, posts_list

        def fetch_comments(self, submission, limit, interval_num):
            """
            Fetch comments from a specific post.

            Args:
                submission: praw.models.reddit.submission.Submission object
                limit: Maximum number of comments to fetch
                interval_num: Interval number for the post

            Returns:
                List of dictionaries containing comment data
            """
            # Sort the comments by best
            submission.comment_sort = 'best'
            submission.comments.replace_more(limit=0)

            # Return the top comments
            return [{'Post/Comment': 'Comment', 'ID': submission.id, 'Text': comment.body,
                        'Creation Date': datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d'),
                        'Interval Number': interval_num} for comment in submission.comments.list()[:limit]]

        def create(self, num_posts, subreddit_name, interval, top_comments_count, output_file):
            """
            Create a DataFrame of posts and comments.

            Args:
                num_posts: Number of posts to fetch
                subreddit_name: Name of the subreddit to fetch posts from
                interval: Time interval to fetch posts from
                top_comments_count: Maximum number of comments to fetch per post
                output_file: File path to save the DataFrame to

            Returns:
                df: pandas.DataFrame containing post and comment data
            """
            # Fetch the posts
            data, posts_list = self.fetch_posts(num_posts, subreddit_name, interval)
            interval_nums = [d['Interval Number'] for d in data]

            # Fetch the comments using a ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor() as executor:
                comments_list = list(executor.map(lambda p: self.fetch_comments(p[0], top_comments_count, p[1]),
                                                list(zip(posts_list, interval_nums))))

            # Extend the data list with the comments
            data.extend([comment for comment_list in comments_list for comment in comment_list])

            # Create a DataFrame from the data
            df = pd.DataFrame(data)
            #df.to_csv(output_file, index=True)
            return df

    # Create an instance of RedditScraper and scrape the data
    scraper = RedditScraper(client_id, client_secret, user_agent)
    tmp = scraper.create(num_posts, subreddit_name, interval, top_comments_count, output_file)
    return tmp