from flask import Flask, jsonify
from jinja2 import escape #pip install Jinja2==3.0.3
import snscrape.modules.twitter as sntwitter
import json
import pandas as pd
import re
import numpy as np
from nltk.tokenize import RegexpTokenizer
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
# HazardType = int(input('HazardType: '))

@app.route("/api/<int:HazardType>", methods=["GET", "POST"])

# hazard types and keywords to scrape tweets for

def TweetScrape(HazardType):
  hazard_types = ["Blizzard", "Sea level rise", "Flood", "Heatwave"]
  hazard_keywords = {"Blizzard":["snowstorm", "freezing"],
                    "Sea level rise": [],
                    "Flood": ["flooding", "river flood", "urban flood"],
                    "Heatwave": ["heatwave", "heat stroke", "heat exhaustion"]
                    }

  #"beach", "global warming"
  # Using TwitterSearchScraper to scrape data and append tweets to list
  def TWTRScrapr(hazard_types):
      tweets_df_list = [] # list to hold the dataframe for each hazard type
      negation_keywords = " -game -movie "
      for hazard_type in hazard_types:
          keywords = ' '.join(hazard_keywords[hazard_type])
          # Created a list to append all tweet attributes(data)
          attributes_container = []
          for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{hazard_type} lang:en {keywords} {negation_keywords} -filter:retweets -filter:replies ').get_items()):
              if i > 10: # limit to 200 tweets
                  break
              if tweet.place:
                  country = tweet.place.country
              else:
                  country = None
              attributes_container.append([tweet.date, tweet.likeCount, tweet.sourceLabel, country, tweet.user.username, tweet.user.followersCount, tweet.content])
          # Creating a dataframe from the tweets list above 
          tweets_df = pd.DataFrame(attributes_container, columns=["Date Created", "Number of Likes", "Source of Tweet", "Country", "Username", "Followers Count", "Tweets"])
          tweets_df_list.append(tweets_df.sort_values(by=['Date Created'], ascending = False).reset_index(drop=True))
      return tweets_df_list

  #This is a list containing 4 dataframes, tweets_df_list[0] for blizzard tweets, tweets_df_list[1] for sea level rise tweets, etc.
  tweets_df_list = TWTRScrapr(hazard_types)

  regexp = RegexpTokenizer('\w+')

  for tweets_df in tweets_df_list:
      tweets_df['text_token'] = tweets_df['Tweets'].apply(regexp.tokenize)

  # Cleaning tweets

  def remove_pattern(text, pattern_regex):
      r = re.findall(pattern_regex, text)
      for i in r:
          text = re.sub(i, '', text)
       
      return text 

  # We are keeping cleaned tweets in a new column called 'Cleaned Tweets'
  for df in tweets_df_list:
      df['Cleaned Tweets'] = np.vectorize(remove_pattern)(df[['Tweets']], r'@\w+')

  from textblob import TextBlob
  from textblob.sentiments import NaiveBayesAnalyzer
  from textblob.np_extractors import ConllExtractor

  # 1 way
  def fetch_sentiment_using_textblob(text):
      analysis = TextBlob(text)
      return 'pos' if analysis.sentiment.polarity >= 0 else 'neg'

  #Applying the function to get sentiment
  for df in tweets_df_list:
      df['Sentiment'] = df['Cleaned Tweets'].apply(fetch_sentiment_using_textblob)
  
  #Cleaned, date created and username
  HazardDF = tweets_df_list[HazardType]

  HazardDF = HazardDF[["Date Created", "Cleaned Tweets","Username","Sentiment"]]
  HazardDF.rename(columns={'Date Created': 'date_created', 'Cleaned Tweets': 'cleaned_tweets','Username':'username','Sentiment':'sentiment'}, inplace=True)
  HazardDF = HazardDF.reset_index(drop = True)

  
  result = HazardDF.to_json(orient='records')


  parsed = json.loads(result)



  json.dumps(parsed, indent=4)  

  return jsonify(parsed)

# TweetScrape(HazardType)

if __name__ == "__main__":
    app.run(debug=True)





