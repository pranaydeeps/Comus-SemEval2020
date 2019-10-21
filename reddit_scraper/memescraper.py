import praw

reddit = praw.Reddit(client_id = 'NMQd5_OLC3nUjw', 
                     client_secret = 'UZIOOn0M8_pR1I-sHTX8Yde6O9w', 
                     user_agent = 'Comus-SemEval2020')

subreddit = self.reddit.subreddit('memes') 
posts = subreddit.top('all')

for post in posts:
  image_urls.append(post.url.encode('utf-8'))
  image_titles.append(post.title.encode('utf-8'))
  image_scores.append(post.score)
  image_timestamps.append(datetime.datetime.fromtimestamp(post.created))
  image_ids.append(post.id)