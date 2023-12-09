import praw
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import prawcore


class RedditScraper:

    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def fetch_posts(self, sub_name, interval):
        subreddit = self.reddit.subreddit(sub_name)
        posts = subreddit.top(limit=None)
        posts_list = list(posts)
        posts_list.sort(key=lambda post: post.created_utc, reverse=True)

        intervals = {
            'daily': timedelta(days=1),
            'weekly': timedelta(weeks=1),
            'monthly': timedelta(weeks=4)
        }

        end_time = datetime.utcfromtimestamp(posts_list[0].created_utc)
        nested_posts = []
        current_interval_start = end_time
        current_interval_posts = []

        for post in posts_list:
            post_time = datetime.utcfromtimestamp(post.created_utc)

            if post_time < current_interval_start - intervals[interval]:
                nested_posts.append(current_interval_posts)
                current_interval_posts = []
                current_interval_start = post_time

            current_interval_posts.append(post)

        if current_interval_posts:
            nested_posts.append(current_interval_posts)

        return nested_posts

    def create_csv(self, nested_posts, top_comments_count, filename='reddit_posts_and_comments.csv'):
        data = []
        total_posts = sum(len(posts) for posts in nested_posts)

        with tqdm(total=total_posts, desc="Processing Posts") as pbar:
            for interval_index, interval_posts in enumerate(nested_posts):
                for post in interval_posts:
                    data.append({
                        'Post/Comment': 'Post',
                        'ID': post.id,
                        'Text': post.title + post.selftext,
                        'Creation Date': datetime.utcfromtimestamp(post.created_utc).strftime('%Y-%m-%d'),
                        'Interval Number': interval_index
                    })

                    try:
                        submission = self.reddit.submission(id=post.id)
                        submission.comment_sort = 'best'
                        submission.comments.replace_more(limit=None)

                        for comment in submission.comments[:top_comments_count]:
                            if hasattr(comment, "author") and comment.author:
                                data.append({
                                    'Post/Comment': 'Comment',
                                    'ID': comment.id,
                                    'Text': comment.body,
                                    'Creation Date': datetime.utcfromtimestamp(comment.created_utc).strftime(
                                        '%Y-%m-%d'),
                                    'Interval Number': interval_index
                                })
                    except praw.exceptions.RedditAPIException as e:
                        print(f"Rate limit exceeded. Waiting for {e.sleep_time} seconds.")
                        time.sleep(e.sleep_time)

                    except prawcore.exceptions.ResponseException as e:

                        if e.response.status_code == 429:
                            retry_after = int(e.response.headers.get('Retry-After', 60))
                            print(f"Rate limit exceeded. Waiting for {retry_after} seconds.")

                    pbar.update(1)

        df = pd.DataFrame(data)
        df.to_csv(filename, index=True)

    def create(self, subreddit_name, interval, top_comments_count, output_file):
        nested_posts = self.fetch_posts(subreddit_name, interval)
        self.create_csv(nested_posts, top_comments_count, output_file)


def main():
    scraper = RedditScraper(client_id="nFKOCvQQEIoW2hFeVG6kfA", client_secret="5BBB4fr-HMPtO8f4jZhle74-fYcDkQ",
                            user_agent="Icy_Process3191")
    scraper.create('pharmacy', 'weekly', 3, 'reddit_posts_and_comments.csv')


if __name__ == "__main__":
    main()
