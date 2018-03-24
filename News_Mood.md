
# News Mood

### Analysis

##### Observed Trend 1: According to the bar chart of the compound VADER sentiment, the sentiment scores of all five news organizations are negative. NY Times has the least negative score. 
##### Observed Trend 2: Visualized from the bar chart of neutral sentiment, the scores across each news organization are pretty close.
##### Observed Trend 3: Non of the news organizations are overwhelmingly positive or negative based on the scatter plot.


```python
# Dependencies
import tweepy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import time
import datetime
from datetime import datetime,tzinfo,timedelta
import json

# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
from twitter_config import consumer_key, consumer_secret, access_token, access_token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
```


```python
# Target User Accounts
target_user = ("@BBCWorld", "@CBSNews", "@CNN", "@FoxNews", "@nytimes")

# Variables for holding sentiments

compound_list = []
positive_list = []
negative_list = []
neutral_list = []

#Additional variables for DataFrame
user_list = []
converted_timestamps = []
tweet_time_ago = []
tweet_text = []
```


```python
# Loop through each user
for user in target_user:
    counter = 0
       
    # Loop through 10 pages of tweets (total 100 tweets)
    for page in tweepy.Cursor(api.user_timeline, id=user).pages(100):

        # Get all tweets from home feed
        page = page[0]
        tweet = json.dumps(page._json, indent=3)
        tweet = json.loads(tweet)
        text = tweet['text']
        raw_time = tweet['created_at']
        
        converted_time = datetime.strptime(raw_time, "%a %b %d %H:%M:%S %z %Y")
        date_stamp = converted_time.strftime("%m-%d-%Y")
        converted_time = converted_time.strftime("%m-%d-%Y %H:%M:%S") 
        
         # Run Vader Analysis on each tweet
        compound = analyzer.polarity_scores(text)["compound"]
        pos = analyzer.polarity_scores(text)["pos"]
        neu = analyzer.polarity_scores(text)["neu"]
        neg = analyzer.polarity_scores(text)["neg"]

        # Add sentiments for each tweet into an array
        user_list.append(user)
        compound_list.append(compound)
        positive_list.append(pos)
        neutral_list.append(neu)
        negative_list.append(neg)
        
        converted_timestamps.append(converted_time)
        tweet_time_ago.append(counter)
        tweet_text.append(text)
        
         # Add to counter 
        counter = counter + 1
        
        if counter % 59 == 0:
            time.sleep(60)
```


```python
# Convert sentiments to DataFrame
twitter_data = {"User" : user_list,
                "Compound" : compound_list,
                "Positive" : positive_list,
                "Neutral" : neutral_list,
                "Negative" : negative_list,
                "Tweet" : tweet_text,
                "Time Stamp" : converted_timestamps,
                "Tweet_Ago" : tweet_time_ago,
               }

twitter_data = pd.DataFrame(twitter_data)

twitter_data.head()
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
      <th>Compound</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Time Stamp</th>
      <th>Tweet</th>
      <th>Tweet_Ago</th>
      <th>User</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>03-24-2018 05:18:00</td>
      <td>French police 'hero' dies of wounds https://t....</td>
      <td>0</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.4939</td>
      <td>0.286</td>
      <td>0.714</td>
      <td>0.0</td>
      <td>03-23-2018 16:14:43</td>
      <td>Car bomb targets spectators at Afghanistan wre...</td>
      <td>1</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>03-23-2018 03:49:46</td>
      <td>RT @BBCNewsAsia: South Korea has some of the l...</td>
      <td>2</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>0.000</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>03-22-2018 19:15:59</td>
      <td>Kenya payout for mother made to deliver on hos...</td>
      <td>3</td>
      <td>@BBCWorld</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.1531</td>
      <td>0.167</td>
      <td>0.833</td>
      <td>0.0</td>
      <td>03-22-2018 13:37:10</td>
      <td>Miss Venezuela to close temporarily over corru...</td>
      <td>4</td>
      <td>@BBCWorld</td>
    </tr>
  </tbody>
</table>
</div>



### Sentiment Analysis of Media Tweets


```python
# Create plot
BBCWorld = twitter_data[(twitter_data.User == "@BBCWorld")]
CBSNews = twitter_data[(twitter_data.User == "@CBSNews")]
CNN = twitter_data[(twitter_data.User == "@CNN")]
FoxNews = twitter_data[(twitter_data.User == "@FoxNews")]
NYTimes = twitter_data[(twitter_data.User == "@nytimes")]

plt.scatter(BBCWorld["Tweet_Ago"], BBCWorld["Compound"], label = "BBC World", marker="o", c=["lightblue"], edgecolors="black")
plt.scatter(CBSNews["Tweet_Ago"], CBSNews["Compound"], label = "CBS News", marker="o", c=["green"], edgecolors="black")
plt.scatter(CNN["Tweet_Ago"], CNN["Compound"], label = "CNN", marker="o", c=["red"], edgecolors="black")
plt.scatter(FoxNews["Tweet_Ago"], FoxNews["Compound"], label = "Fox News", marker="o", c=["blue"], edgecolors="black")
plt.scatter(NYTimes["Tweet_Ago"], NYTimes["Compound"], label = "NY Times", marker="o", c=["yellow"], edgecolors="black")

# Plot field size
plt.xlim(-5, counter + 5)
plt.ylim(-1.25, 1.25)

# Incorporate the other graph properties
plt.title("Sentiment Analysis of Media Tweets" + " " + str(date_stamp))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid(True)

plt.legend(title="Media Sources", bbox_to_anchor=(1.25, 1))

# Save the figure
plt.savefig("Sentiment Analysis of Media Tweets.png")

# invert x axis
ax = plt.gca()
ax.invert_xaxis()

# Show plot
plt.show()
```


![png](output_7_0.png)


### Overall Media Sentiment based on Twitter


```python
# Create plot
BBCWorld = twitter_data[(twitter_data.User == "@BBCWorld")]
CBSNews = twitter_data[(twitter_data.User == "@CBSNews")]
CNN = twitter_data[(twitter_data.User == "@CNN")]
FoxNews = twitter_data[(twitter_data.User == "@FoxNews")]
NYTimes = twitter_data[(twitter_data.User == "@nytimes")]

BBCWorld_average_compound = BBCWorld.mean()["Compound"]
CBSNews_average_compound = CBSNews.mean()["Compound"]
CNN_average_compound = CNN.mean()["Compound"]
FoxNews_average_compound = FoxNews.mean()["Compound"]
NYTimes_average_compound = NYTimes.mean()["Compound"]

# Create an array that contains the number of users each language has
users = [BBCWorld_average_compound, CBSNews_average_compound, CNN_average_compound, FoxNews_average_compound, NYTimes_average_compound]
x_axis = np.arange(len(users))

# Tell matplotlib that we will be making a bar chart
plt.bar(x_axis, users, width=1, color=["lightblue", "green", "red", "blue", "yellow"], alpha=0.5, align="edge")

# Tell matplotlib where we would like to place each of our x axis headers
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, ["BBC World", "CBS News", "CNN", "Fox News", "NY Times"])

# Plot field size
plt.ylim(-.25, .25)

# Incorporate the other graph properties
plt.title("Overall Media Sentiment Based on Twitter, Compound Sentiment" + " " + str(date_stamp))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid(False)

# # Save the figure
plt.savefig("Overall Media Sentiment Based on Twitter, Compound Sentiment.png")

# Show plot
plt.show()
```


![png](output_9_0.png)



```python
# Create plot
BBCWorld = twitter_data[(twitter_data.User == "@BBCWorld")]
CBSNews = twitter_data[(twitter_data.User == "@CBSNews")]
CNN = twitter_data[(twitter_data.User == "@CNN")]
FoxNews = twitter_data[(twitter_data.User == "@FoxNews")]
NYTimes = twitter_data[(twitter_data.User == "@nytimes")]

BBCWorld_average_positive = BBCWorld.mean()["Positive"]
CBSNews_average_positive = CBSNews.mean()["Positive"]
CNN_average_positive = CNN.mean()["Positive"]
FoxNews_average_positive = FoxNews.mean()["Positive"]
NYTimes_average_positive = NYTimes.mean()["Positive"]

# Create an array that contains the number of users each language has
users = [BBCWorld_average_positive, CBSNews_average_positive, CNN_average_positive, FoxNews_average_positive, NYTimes_average_positive]
x_axis = np.arange(len(users))

# Tell matplotlib that we will be making a bar chart
plt.bar(x_axis, users, width=1, color=["lightblue", "green", "red", "blue", "yellow"], alpha=0.5, align="edge")

# Tell matplotlib where we would like to place each of our x axis headers
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, ["BBC World", "CBS News", "CNN", "Fox News", "NY Times"])

# Plot field size
plt.ylim(-.25, .25)

# Incorporate the other graph properties
plt.title("Overall Media Sentiment Based on Twitter, Positive Sentiment" + " " + str(date_stamp))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid(False)

# # Save the figure
plt.savefig("Overall Media Sentiment Based on Twitter, Positive Sentiment.png")

# Show plot
plt.show()
```


![png](output_10_0.png)



```python
# Create plot
BBCWorld = twitter_data[(twitter_data.User == "@BBCWorld")]
CBSNews = twitter_data[(twitter_data.User == "@CBSNews")]
CNN = twitter_data[(twitter_data.User == "@CNN")]
FoxNews = twitter_data[(twitter_data.User == "@FoxNews")]
NYTimes = twitter_data[(twitter_data.User == "@nytimes")]

BBCWorld_average_negative = BBCWorld.mean()["Negative"]
CBSNews_average_negative = CBSNews.mean()["Negative"]
CNN_average_negative = CNN.mean()["Negative"]
FoxNews_average_negative = FoxNews.mean()["Negative"]
NYTimes_average_negative = NYTimes.mean()["Negative"]

# Create an array that contains the number of users each language has
users = [BBCWorld_average_negative, CBSNews_average_negative, CNN_average_negative, FoxNews_average_negative, NYTimes_average_negative]
x_axis = np.arange(len(users))

# Tell matplotlib that we will be making a bar chart
plt.bar(x_axis, users, width=1, color=["lightblue", "green", "red", "blue", "yellow"], alpha=0.5, align="edge")

# Tell matplotlib where we would like to place each of our x axis headers
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, ["BBC World", "CBS News", "CNN", "Fox News", "NY Times"])

# Plot field size
plt.ylim(-.25, .25)

# Incorporate the other graph properties
plt.title("Overall Media Sentiment Based on Twitter, Negative Sentiment" + " " + str(date_stamp))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid(False)

# invert y axis, positive score reflects a negative responce
ax = plt.gca()
ax.invert_yaxis()

# # Save the figure
plt.savefig("Overall Media Sentiment Based on Twitter, Negative Sentiment.png")

# Show plot
plt.show()
```


![png](output_11_0.png)



```python
# Create plot
BBCWorld = twitter_data[(twitter_data.User == "@BBCWorld")]
CBSNews = twitter_data[(twitter_data.User == "@CBSNews")]
CNN = twitter_data[(twitter_data.User == "@CNN")]
FoxNews = twitter_data[(twitter_data.User == "@FoxNews")]
NYTimes = twitter_data[(twitter_data.User == "@nytimes")]

BBCWorld_average_neutral = BBCWorld.mean()["Neutral"]
CBSNews_average_neutral = CBSNews.mean()["Neutral"]
CNN_average_neutral = CNN.mean()["Neutral"]
FoxNews_average_neutral = FoxNews.mean()["Neutral"]
NYTimes_average_neutral = NYTimes.mean()["Neutral"]

# Create an array that contains the number of users each language has
users = [BBCWorld_average_neutral, CBSNews_average_neutral, CNN_average_neutral, FoxNews_average_neutral, NYTimes_average_neutral]
x_axis = np.arange(len(users))

# Tell matplotlib that we will be making a bar chart
plt.bar(x_axis, users, width=1, color=["lightblue", "green", "red", "blue", "yellow"], alpha=0.5, align="edge")

# Tell matplotlib where we would like to place each of our x axis headers
tick_locations = [value+0.4 for value in x_axis]
plt.xticks(tick_locations, ["BBC World", "CBS News", "CNN", "Fox News", "NY Times"])

# Plot field size
plt.ylim(-1, 1)

# Incorporate the other graph properties
plt.title("Overall Media Sentiment Based on Twitter, Neutral Sentiment" + " " + str(date_stamp))
plt.ylabel("Tweet Polarity")
plt.xlabel("Tweets Ago")
plt.grid(False)

# # Save the figure
plt.savefig("Overall Media Sentiment Based on Twitter, Neutral Sentiment.png")

# Show plot
plt.show()
```


![png](output_12_0.png)



```python
# Export the data in the DataFrame into a CSV file
twitter_data.to_csv("Twitter_Data.csv")
```
