import pandas as pd


#df = pd.read_json('reddit_posts_and_comments.jsonl', lines=True)

df = pd.read_csv('reddit_posts_and_comments.csv', parse_dates = ['Creation Date'])
#d = df.set_index('Creation Date')
#print(df)
g = df.groupby(pd.Grouper(key='Creation Date', freq='M'))
dfs = [group for _,group in g]
print(dfs[3])