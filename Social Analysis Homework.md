

```python
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
```


```python
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
```


```python
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
```


```python
df_tweet_data = pd.DataFrame.from_dict(tweet_data)
df_tweet_data = df_tweet_data.reset_index()
df_tweet_data = df_tweet_data.rename(columns={'index':'tweet_count'})
df_tweet_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_count</th>
      <th>tweet_comp_score</th>
      <th>tweet_date</th>
      <th>tweet_neg_score</th>
      <th>tweet_neu_score</th>
      <th>tweet_pos_score</th>
      <th>tweet_source</th>
      <th>tweet_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.5574</td>
      <td>Tue Jul 10 07:01:05 +0000 2018</td>
      <td>0.073</td>
      <td>0.733</td>
      <td>0.193</td>
      <td>BBC News (World)</td>
      <td>RT @SallyBundockBBC: The great escape - if you...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>-0.7906</td>
      <td>Tue Jul 10 06:58:49 +0000 2018</td>
      <td>0.467</td>
      <td>0.533</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Syria war: What we know about Douma 'chemical ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.4404</td>
      <td>Tue Jul 10 06:12:46 +0000 2018</td>
      <td>0.000</td>
      <td>0.707</td>
      <td>0.293</td>
      <td>BBC News (World)</td>
      <td>World's 'oldest coloured molecules' are bright...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.5106</td>
      <td>Tue Jul 10 06:10:20 +0000 2018</td>
      <td>0.000</td>
      <td>0.732</td>
      <td>0.268</td>
      <td>BBC News (World)</td>
      <td>The latest from the cave rescue in Thailand\n\...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>-0.0516</td>
      <td>Tue Jul 10 05:41:00 +0000 2018</td>
      <td>0.130</td>
      <td>0.870</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Dissident Liu Xiaobo's widow 'allowed to leave...</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>0.5106</td>
      <td>Tue Jul 10 05:40:21 +0000 2018</td>
      <td>0.000</td>
      <td>0.798</td>
      <td>0.202</td>
      <td>BBC News (World)</td>
      <td>Elon Musk's offer 'not practical' for cave mis...</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>0.7184</td>
      <td>Tue Jul 10 05:15:28 +0000 2018</td>
      <td>0.000</td>
      <td>0.727</td>
      <td>0.273</td>
      <td>BBC News (World)</td>
      <td>RT @BBCBreaking: #ThailandCaveRescue: Remainin...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>-0.6705</td>
      <td>Tue Jul 10 02:55:20 +0000 2018</td>
      <td>0.333</td>
      <td>0.667</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Japan floods: 126 killed after torrential rain...</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>0.0258</td>
      <td>Tue Jul 10 02:47:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.952</td>
      <td>0.048</td>
      <td>BBC News (World)</td>
      <td>RT @SallyBundockBBC: The #WorldCup is down to ...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>-0.4215</td>
      <td>Tue Jul 10 02:40:04 +0000 2018</td>
      <td>0.237</td>
      <td>0.763</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Australia and NZ recall frozen vegetables over...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>0.0000</td>
      <td>Tue Jul 10 02:25:56 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Follow our live #ThailandCaveRescue page for t...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>0.5574</td>
      <td>Tue Jul 10 01:07:55 +0000 2018</td>
      <td>0.000</td>
      <td>0.660</td>
      <td>0.340</td>
      <td>BBC News (World)</td>
      <td>Trump names Kavanaugh for US Supreme Court htt...</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>0.5423</td>
      <td>Mon Jul 09 23:33:17 +0000 2018</td>
      <td>0.000</td>
      <td>0.741</td>
      <td>0.259</td>
      <td>BBC News (World)</td>
      <td>Can Ethiopia's Abiy Ahmed make peace with 'Afr...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>0.0000</td>
      <td>Mon Jul 09 23:28:17 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Macedonia: The Balkan country waiting for Nato...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>-0.2732</td>
      <td>Mon Jul 09 23:11:16 +0000 2018</td>
      <td>0.259</td>
      <td>0.741</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Where a six-figure salary is 'low income' http...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>0.4404</td>
      <td>Mon Jul 09 23:08:26 +0000 2018</td>
      <td>0.000</td>
      <td>0.707</td>
      <td>0.293</td>
      <td>BBC News (World)</td>
      <td>Belgium-France: World Cup clash tests border l...</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>0.0000</td>
      <td>Mon Jul 09 21:44:58 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Turkey's Erdogan appoints son-in-law as financ...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>0.8402</td>
      <td>Mon Jul 09 21:42:09 +0000 2018</td>
      <td>0.000</td>
      <td>0.438</td>
      <td>0.562</td>
      <td>BBC News (World)</td>
      <td>Cave rescue: Divers ready to save remaining fi...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>-0.8225</td>
      <td>Mon Jul 09 21:34:47 +0000 2018</td>
      <td>0.487</td>
      <td>0.513</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Nicaragua unrest: Bishop's anger as people die...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>-0.5574</td>
      <td>Mon Jul 09 20:58:04 +0000 2018</td>
      <td>0.265</td>
      <td>0.735</td>
      <td>0.000</td>
      <td>BBC News (World)</td>
      <td>Starbucks to ban plastics straws in all stores...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>0.0000</td>
      <td>Tue Jul 10 07:03:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBS News</td>
      <td>Build-A-Bear Workshop to let age determine cos...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>0.3724</td>
      <td>Tue Jul 10 06:48:04 +0000 2018</td>
      <td>0.000</td>
      <td>0.825</td>
      <td>0.175</td>
      <td>CBS News</td>
      <td>Mayor: Officer seen Tasing man sitting on curb...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>0.4215</td>
      <td>Tue Jul 10 06:33:03 +0000 2018</td>
      <td>0.000</td>
      <td>0.833</td>
      <td>0.167</td>
      <td>CBS News</td>
      <td>Russia launches Progress supply ship on fastes...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>-0.3612</td>
      <td>Tue Jul 10 06:18:03 +0000 2018</td>
      <td>0.217</td>
      <td>0.783</td>
      <td>0.000</td>
      <td>CBS News</td>
      <td>Former FBI lawyer Lisa Page fighting congressi...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>0.0000</td>
      <td>Tue Jul 10 06:03:03 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBS News</td>
      <td>Reactions to Trump's decision to nominate Bret...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>0.0000</td>
      <td>Tue Jul 10 06:00:26 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBS News</td>
      <td>Trump vowed to nominate justices who would ove...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>0.0000</td>
      <td>Tue Jul 10 05:48:04 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>CBS News</td>
      <td>"This is what we call a miracle": Baby survive...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>0.5574</td>
      <td>Tue Jul 10 05:33:03 +0000 2018</td>
      <td>0.000</td>
      <td>0.769</td>
      <td>0.231</td>
      <td>CBS News</td>
      <td>Nuclear option: Why Trump's Supreme Court pick...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>-0.7783</td>
      <td>Tue Jul 10 05:18:04 +0000 2018</td>
      <td>0.343</td>
      <td>0.657</td>
      <td>0.000</td>
      <td>CBS News</td>
      <td>China makes $8.46 from an iPhone. That's why a...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>0.4767</td>
      <td>Tue Jul 10 05:04:29 +0000 2018</td>
      <td>0.126</td>
      <td>0.632</td>
      <td>0.242</td>
      <td>CBS News</td>
      <td>HAPPENING NOW: Third rescue mission is now und...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>470</th>
      <td>470</td>
      <td>-0.3818</td>
      <td>Mon Jul 09 23:10:10 +0000 2018</td>
      <td>0.271</td>
      <td>0.729</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>SCOTUS pick will trigger confirmation battle h...</td>
    </tr>
    <tr>
      <th>471</th>
      <td>471</td>
      <td>0.4019</td>
      <td>Mon Jul 09 23:05:00 +0000 2018</td>
      <td>0.000</td>
      <td>0.863</td>
      <td>0.137</td>
      <td>Fox News</td>
      <td>On @foxandfriends, Rudy Giuliani sounded off o...</td>
    </tr>
    <tr>
      <th>472</th>
      <td>472</td>
      <td>-0.4404</td>
      <td>Mon Jul 09 22:56:17 +0000 2018</td>
      <td>0.209</td>
      <td>0.791</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>Boy, 8, gets 18 doses of anti-venom after pain...</td>
    </tr>
    <tr>
      <th>473</th>
      <td>473</td>
      <td>-0.7906</td>
      <td>Mon Jul 09 22:50:56 +0000 2018</td>
      <td>0.412</td>
      <td>0.588</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>2 dead after California train, car crash that ...</td>
    </tr>
    <tr>
      <th>474</th>
      <td>474</td>
      <td>0.5574</td>
      <td>Mon Jul 09 22:44:28 +0000 2018</td>
      <td>0.000</td>
      <td>0.769</td>
      <td>0.231</td>
      <td>Fox News</td>
      <td>.@NYGovCuomo signs reproductive rights executi...</td>
    </tr>
    <tr>
      <th>475</th>
      <td>475</td>
      <td>-0.8271</td>
      <td>Mon Jul 09 22:41:00 +0000 2018</td>
      <td>0.367</td>
      <td>0.633</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>.@stephenfhayes on the battle over the #SCOTUS...</td>
    </tr>
    <tr>
      <th>476</th>
      <td>476</td>
      <td>-0.4767</td>
      <td>Mon Jul 09 22:35:04 +0000 2018</td>
      <td>0.181</td>
      <td>0.819</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>Tropical Storm Chris expected to become hurric...</td>
    </tr>
    <tr>
      <th>477</th>
      <td>477</td>
      <td>-0.5106</td>
      <td>Mon Jul 09 22:28:51 +0000 2018</td>
      <td>0.292</td>
      <td>0.708</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>Authorities investigate Jeep Wrangler burned i...</td>
    </tr>
    <tr>
      <th>478</th>
      <td>478</td>
      <td>0.0000</td>
      <td>Mon Jul 09 22:24:14 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>A video taken of a heated situation in El Paso...</td>
    </tr>
    <tr>
      <th>479</th>
      <td>479</td>
      <td>0.0000</td>
      <td>Mon Jul 09 22:17:12 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>Fox News</td>
      <td>Starbucks says it's ditching all plastic straw...</td>
    </tr>
    <tr>
      <th>480</th>
      <td>480</td>
      <td>0.0000</td>
      <td>Mon Jul 09 12:16:31 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>Morning Briefing: Here's what you need to know...</td>
    </tr>
    <tr>
      <th>481</th>
      <td>481</td>
      <td>0.0000</td>
      <td>Mon Jul 09 12:04:41 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>On StockX, sneakers, streetwear, handbags and ...</td>
    </tr>
    <tr>
      <th>482</th>
      <td>482</td>
      <td>0.1531</td>
      <td>Mon Jul 09 11:46:18 +0000 2018</td>
      <td>0.133</td>
      <td>0.708</td>
      <td>0.158</td>
      <td>The New York Times</td>
      <td>RT @nytimestravel: 36 Hours in Tbilisi: With c...</td>
    </tr>
    <tr>
      <th>483</th>
      <td>483</td>
      <td>-0.5106</td>
      <td>Mon Jul 09 11:28:36 +0000 2018</td>
      <td>0.148</td>
      <td>0.852</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>A judge in Brazil ruled that former President ...</td>
    </tr>
    <tr>
      <th>484</th>
      <td>484</td>
      <td>-0.4215</td>
      <td>Mon Jul 09 11:10:27 +0000 2018</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>Amid Japan’s Flood Devastation, Survivors Dig ...</td>
    </tr>
    <tr>
      <th>485</th>
      <td>485</td>
      <td>0.0000</td>
      <td>Mon Jul 09 11:08:01 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>RT @nytimesbusiness: Sports teams, leagues and...</td>
    </tr>
    <tr>
      <th>486</th>
      <td>486</td>
      <td>0.2500</td>
      <td>Mon Jul 09 10:51:22 +0000 2018</td>
      <td>0.000</td>
      <td>0.905</td>
      <td>0.095</td>
      <td>The New York Times</td>
      <td>In her new staging of T. S. Eliot’s poems “Fou...</td>
    </tr>
    <tr>
      <th>487</th>
      <td>487</td>
      <td>0.2500</td>
      <td>Mon Jul 09 10:33:49 +0000 2018</td>
      <td>0.000</td>
      <td>0.667</td>
      <td>0.333</td>
      <td>The New York Times</td>
      <td>Deporting the American Dream https://t.co/pDKi...</td>
    </tr>
    <tr>
      <th>488</th>
      <td>488</td>
      <td>0.0000</td>
      <td>Mon Jul 09 10:33:40 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>For many Egyptians, the price of a career in b...</td>
    </tr>
    <tr>
      <th>489</th>
      <td>489</td>
      <td>0.5106</td>
      <td>Mon Jul 09 10:16:50 +0000 2018</td>
      <td>0.000</td>
      <td>0.732</td>
      <td>0.268</td>
      <td>The New York Times</td>
      <td>Bites: A Pittsburgh Area Restaurant That Trade...</td>
    </tr>
    <tr>
      <th>490</th>
      <td>490</td>
      <td>0.8176</td>
      <td>Mon Jul 09 10:14:49 +0000 2018</td>
      <td>0.000</td>
      <td>0.622</td>
      <td>0.378</td>
      <td>The New York Times</td>
      <td>A resolution to encourage breast-feeding was e...</td>
    </tr>
    <tr>
      <th>491</th>
      <td>491</td>
      <td>0.2500</td>
      <td>Mon Jul 09 09:58:03 +0000 2018</td>
      <td>0.095</td>
      <td>0.762</td>
      <td>0.143</td>
      <td>The New York Times</td>
      <td>Tom Wolfe had a lesser-known passion: drawing ...</td>
    </tr>
    <tr>
      <th>492</th>
      <td>492</td>
      <td>0.0000</td>
      <td>Mon Jul 09 09:40:11 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>Japanese officials pleaded with millions of pe...</td>
    </tr>
    <tr>
      <th>493</th>
      <td>493</td>
      <td>0.6697</td>
      <td>Mon Jul 09 09:23:31 +0000 2018</td>
      <td>0.000</td>
      <td>0.703</td>
      <td>0.297</td>
      <td>The New York Times</td>
      <td>Tech tools are far more efficient at keeping y...</td>
    </tr>
    <tr>
      <th>494</th>
      <td>494</td>
      <td>0.7003</td>
      <td>Mon Jul 09 09:06:18 +0000 2018</td>
      <td>0.000</td>
      <td>0.595</td>
      <td>0.405</td>
      <td>The New York Times</td>
      <td>Do You Like ‘Dogs Playing Poker’? Science Woul...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>495</td>
      <td>0.5106</td>
      <td>Mon Jul 09 09:03:22 +0000 2018</td>
      <td>0.000</td>
      <td>0.883</td>
      <td>0.117</td>
      <td>The New York Times</td>
      <td>RT @nytimesworld: A new rescue run is underway...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>496</td>
      <td>0.0000</td>
      <td>Mon Jul 09 08:51:34 +0000 2018</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>The New York Times</td>
      <td>Once a rising teen pop star herself, Teddy Gei...</td>
    </tr>
    <tr>
      <th>497</th>
      <td>497</td>
      <td>0.8689</td>
      <td>Mon Jul 09 08:33:28 +0000 2018</td>
      <td>0.000</td>
      <td>0.426</td>
      <td>0.574</td>
      <td>The New York Times</td>
      <td>Thailand Cave Rescue Live Updates: Rescued Boy...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>498</td>
      <td>0.7096</td>
      <td>Mon Jul 09 08:25:24 +0000 2018</td>
      <td>0.000</td>
      <td>0.670</td>
      <td>0.330</td>
      <td>The New York Times</td>
      <td>Don't be a bridezilla. Here are a few ways to ...</td>
    </tr>
    <tr>
      <th>499</th>
      <td>499</td>
      <td>0.7430</td>
      <td>Mon Jul 09 08:08:43 +0000 2018</td>
      <td>0.082</td>
      <td>0.599</td>
      <td>0.318</td>
      <td>The New York Times</td>
      <td>“There were no support groups for people like ...</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 8 columns</p>
</div>




```python
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
```


![png](output_4_0.png)



    <matplotlib.figure.Figure at 0x116666a58>



```python
df_tweet_mean = np.round(df_tweet_data.groupby('tweet_source').mean(),3)
df_tweet_mean = df_tweet_mean.drop(columns=['tweet_count','tweet_neg_score','tweet_neu_score','tweet_pos_score'])
df_tweet_comp = df_tweet_mean.reset_index()
df_tweet_comp
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tweet_source</th>
      <th>tweet_comp_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BBC News (World)</td>
      <td>0.007</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CBS News</td>
      <td>0.053</td>
    </tr>
    <tr>
      <th>2</th>
      <td>CNN</td>
      <td>0.102</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Fox News</td>
      <td>0.211</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The New York Times</td>
      <td>0.070</td>
    </tr>
  </tbody>
</table>
</div>




```python
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
```


![png](output_6_0.png)



    <matplotlib.figure.Figure at 0x1a20924710>



```python
df_tweet_data.to_csv('tweet_data.csv')
```
