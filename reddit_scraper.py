import praw
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
import prawcore
from concurrent.futures import ThreadPoolExecutor


class RedditScraper:

    def __init__(self, client_id, client_secret, user_agent):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )

    def fetch_posts(self, num_posts, sub_name, interval):
        subreddit = self.reddit.subreddit(sub_name)
        posts = subreddit.top(time_filter='week', limit=num_posts)
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
        data = []
        interval_num = 0

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
                'Interval Number': interval_num
            })

        return data, posts_list

    def fetch_comments(self, submission, limit, interval_num):
        submission.comment_sort = 'best'
        submission.comments.replace_more(limit=0)

        return [{'Post/Comment': 'Comment', 'ID': submission.id, 'Text': comment.body,
                 'Creation Date': datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d'),
                 'Interval Number': interval_num} for comment in submission.comments.list()[:limit]]

    def create(self, num_posts, subreddit_name, interval, top_comments_count, output_file):

        data, posts_list = self.fetch_posts(num_posts, subreddit_name, interval)
        interval_nums = [d['Interval Number'] for d in data]

        with ThreadPoolExecutor() as executor:
            comments_list = list(executor.map(lambda p: self.fetch_comments(p[0], top_comments_count, p[1]),
                                              list(zip(posts_list, interval_nums))))

        data.extend([comment for comment_list in comments_list for comment in comment_list])

        print(len(data))
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=True)


def main():
    scraper = RedditScraper(client_id="nFKOCvQQEIoW2hFeVG6kfA", client_secret="5BBB4fr-HMPtO8f4jZhle74-fYcDkQ",
                            user_agent="Icy_Process3191")
    scraper.create(10, 'natureporn', 'weekly', 3, 'reddit_posts_and_comments.csv')


if __name__ == "__main__":
    main()
