
# coding: utf-8

# In[1]:


import tweepy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

from twitter_config import (consumer_key,
                    consumer_key_secret,
                    access_token,
                    access_token_secret)

auth = tweepy.OAuthHandler(consumer_key, consumer_key_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())


# In[2]:


sentiments=[]
compound_list = []
positive_list = []
negative_list = []
neutral_list = []

tweet_data = {
    "tweet_source": [],
    "tweet_text": [],
    "tweet_date": [],
    "tweet_comp_score": [],
    "tweet_neg_score": [],
    "tweet_pos_score": [],
    "tweet_neu_score": []
}


# In[3]:


for x in range(1,6):
    target_user = ("@BBCWorld", "@CBSNews", "@CNN", "@FoxNews", "@nytimes")
    
    for user in target_user:
        public_tweets = api.user_timeline(user, page=x)

        for tweet in public_tweets:
            tweet_data["tweet_source"].append(tweet["user"]["name"])
            tweet_data["tweet_text"].append(tweet["text"])
            tweet_data["tweet_date"].append(tweet["created_at"])
            tweet_data["tweet_comp_score"].append(analyzer.polarity_scores(tweet["text"])["compound"])
            tweet_data["tweet_neg_score"].append(analyzer.polarity_scores(tweet["text"])["neg"])
            tweet_data["tweet_pos_score"].append(analyzer.polarity_scores(tweet["text"])["pos"])
            tweet_data["tweet_neu_score"].append(analyzer.polarity_scores(tweet["text"])["neu"])


# In[4]:


df_tweet_data = pd.DataFrame.from_dict(tweet_data)
df_tweet_data = df_tweet_data.reset_index()
df_tweet_data = df_tweet_data.rename(columns={'index':'tweet_count'})
df_tweet_data


# In[5]:


ax = sns.lmplot(x='tweet_count',
            y='tweet_comp_score',
            data=df_tweet_data,
            size=7,
            aspect=2,
            hue='tweet_source',
            legend=False)
ax.add_legend(label_order=['BBC News (World)','CBS News','CNN','Fox News','The New York Times'])
plt.title('Sentiment Analysis of Media Tweets',fontsize=24,fontweight='bold')
plt.xlabel('Tweets Ago',fontsize=18,fontweight='bold')
plt.ylabel('Tweet Polarity',fontsize=18,fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.show()
plt.savefig('sentiment_analysis.png')


# In[6]:


df_tweet_mean = np.round(df_tweet_data.groupby('tweet_source').mean(),3)
df_tweet_mean = df_tweet_mean.drop(columns=['tweet_count','tweet_neg_score','tweet_neu_score','tweet_pos_score'])
df_tweet_comp = df_tweet_mean.reset_index()
df_tweet_comp


# In[7]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
ax = sns.barplot(x='tweet_source',
            y='tweet_comp_score',
            data=df_tweet_comp)
plt.title('Overall Media Sentiment Based On Twitter',fontsize=24,fontweight='bold')
plt.xlabel('News Organizations',fontsize=18,fontweight='bold')
plt.ylabel('Tweet Polarity',fontsize=18,fontweight='bold')
plt.xticks(fontsize=16, fontweight='bold')
plt.yticks(fontsize=16, fontweight='bold')
plt.show()
plt.savefig('compound_sentiment.png')


# In[8]:


df_tweet_data.to_csv('tweet_data.csv')

