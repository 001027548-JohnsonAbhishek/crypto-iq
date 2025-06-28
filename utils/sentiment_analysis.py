import pandas as pd
import numpy as np
import requests
import tweepy
import praw
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from datetime import datetime, timedelta
import time

class SentimentAnalyzer:
    def __init__(self):
        self.twitter_api = None
        self.reddit_api = None
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.setup_apis()
    
    def setup_apis(self):
        """Setup social media APIs"""
        try:
            # Twitter API setup
            twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN', '')
            if twitter_bearer_token:
                self.twitter_api = tweepy.Client(bearer_token=twitter_bearer_token)
            
            # Reddit API setup
            reddit_client_id = os.getenv('REDDIT_CLIENT_ID', '')
            reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', '')
            reddit_user_agent = os.getenv('REDDIT_USER_AGENT', 'CryptoAnalyzer')
            
            if reddit_client_id and reddit_client_secret:
                self.reddit_api = praw.Reddit(
                    client_id=reddit_client_id,
                    client_secret=reddit_client_secret,
                    user_agent=reddit_user_agent
                )
        except Exception as e:
            print(f"API setup error: {str(e)}")
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of a single text using multiple methods"""
        try:
            # TextBlob analysis
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            textblob_subjectivity = blob.sentiment.subjectivity
            
            # VADER analysis
            vader_scores = self.vader_analyzer.polarity_scores(text)
            
            return {
                'textblob_polarity': textblob_polarity,
                'textblob_subjectivity': textblob_subjectivity,
                'vader_compound': vader_scores['compound'],
                'vader_positive': vader_scores['pos'],
                'vader_negative': vader_scores['neg'],
                'vader_neutral': vader_scores['neu']
            }
        except Exception as e:
            print(f"Text sentiment analysis error: {str(e)}")
            return None
    
    def get_twitter_sentiment(self, cryptocurrency, max_tweets=100):
        """Get sentiment from Twitter"""
        try:
            if not self.twitter_api:
                return None
            
            # Search for tweets
            query = f"{cryptocurrency} OR #{cryptocurrency.replace('-', '').lower()} -is:retweet lang:en"
            tweets = tweepy.Paginator(
                self.twitter_api.search_recent_tweets,
                query=query,
                max_results=min(max_tweets, 100)
            ).flatten(limit=max_tweets)
            
            sentiments = []
            tweet_data = []
            
            for tweet in tweets:
                if tweet.text:
                    sentiment = self.analyze_text_sentiment(tweet.text)
                    if sentiment:
                        sentiments.append(sentiment)
                        tweet_data.append({
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'sentiment': sentiment
                        })
            
            if not sentiments:
                return None
            
            # Calculate aggregate sentiment
            avg_sentiment = {
                'textblob_polarity': np.mean([s['textblob_polarity'] for s in sentiments]),
                'textblob_subjectivity': np.mean([s['textblob_subjectivity'] for s in sentiments]),
                'vader_compound': np.mean([s['vader_compound'] for s in sentiments]),
                'vader_positive': np.mean([s['vader_positive'] for s in sentiments]),
                'vader_negative': np.mean([s['vader_negative'] for s in sentiments]),
                'vader_neutral': np.mean([s['vader_neutral'] for s in sentiments]),
                'total_tweets': len(sentiments)
            }
            
            return {
                'aggregate_sentiment': avg_sentiment,
                'individual_tweets': tweet_data
            }
            
        except Exception as e:
            print(f"Twitter sentiment analysis error: {str(e)}")
            return None
    
    def get_reddit_sentiment(self, cryptocurrency, subreddits=['CryptoCurrency', 'Bitcoin', 'ethereum'], max_posts=50):
        """Get sentiment from Reddit"""
        try:
            if not self.reddit_api:
                return None
            
            all_posts = []
            sentiments = []
            
            for subreddit_name in subreddits:
                try:
                    subreddit = self.reddit_api.subreddit(subreddit_name)
                    
                    # Search for posts mentioning the cryptocurrency
                    search_term = cryptocurrency.replace('-USD', '').replace('-', '')
                    posts = subreddit.search(search_term, limit=max_posts//len(subreddits), time_filter='week')
                    
                    for post in posts:
                        if post.selftext or post.title:
                            text = f"{post.title} {post.selftext}"
                            sentiment = self.analyze_text_sentiment(text)
                            
                            if sentiment:
                                sentiments.append(sentiment)
                                all_posts.append({
                                    'title': post.title,
                                    'text': post.selftext,
                                    'score': post.score,
                                    'created_utc': post.created_utc,
                                    'sentiment': sentiment,
                                    'subreddit': subreddit_name
                                })
                
                except Exception as e:
                    print(f"Error processing subreddit {subreddit_name}: {str(e)}")
                    continue
            
            if not sentiments:
                return None
            
            # Calculate aggregate sentiment
            avg_sentiment = {
                'textblob_polarity': np.mean([s['textblob_polarity'] for s in sentiments]),
                'textblob_subjectivity': np.mean([s['textblob_subjectivity'] for s in sentiments]),
                'vader_compound': np.mean([s['vader_compound'] for s in sentiments]),
                'vader_positive': np.mean([s['vader_positive'] for s in sentiments]),
                'vader_negative': np.mean([s['vader_negative'] for s in sentiments]),
                'vader_neutral': np.mean([s['vader_neutral'] for s in sentiments]),
                'total_posts': len(sentiments)
            }
            
            return {
                'aggregate_sentiment': avg_sentiment,
                'individual_posts': all_posts
            }
            
        except Exception as e:
            print(f"Reddit sentiment analysis error: {str(e)}")
            return None
    
    def get_news_sentiment(self, cryptocurrency):
        """Get sentiment from news articles"""
        try:
            # Using News API
            news_api_key = os.getenv('NEWS_API_KEY', '')
            if not news_api_key:
                return None
            
            # Search for news articles
            search_term = cryptocurrency.replace('-USD', '').replace('-', ' ')
            url = f"https://newsapi.org/v2/everything"
            params = {
                'q': f"{search_term} cryptocurrency",
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': news_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code != 200:
                return None
            
            articles = response.json().get('articles', [])
            
            sentiments = []
            article_data = []
            
            for article in articles:
                if article.get('title') and article.get('description'):
                    text = f"{article['title']} {article['description']}"
                    sentiment = self.analyze_text_sentiment(text)
                    
                    if sentiment:
                        sentiments.append(sentiment)
                        article_data.append({
                            'title': article['title'],
                            'description': article['description'],
                            'url': article['url'],
                            'published_at': article['publishedAt'],
                            'source': article['source']['name'],
                            'sentiment': sentiment
                        })
            
            if not sentiments:
                return None
            
            # Calculate aggregate sentiment
            avg_sentiment = {
                'textblob_polarity': np.mean([s['textblob_polarity'] for s in sentiments]),
                'textblob_subjectivity': np.mean([s['textblob_subjectivity'] for s in sentiments]),
                'vader_compound': np.mean([s['vader_compound'] for s in sentiments]),
                'vader_positive': np.mean([s['vader_positive'] for s in sentiments]),
                'vader_negative': np.mean([s['vader_negative'] for s in sentiments]),
                'vader_neutral': np.mean([s['vader_neutral'] for s in sentiments]),
                'total_articles': len(sentiments)
            }
            
            return {
                'aggregate_sentiment': avg_sentiment,
                'individual_articles': article_data
            }
            
        except Exception as e:
            print(f"News sentiment analysis error: {str(e)}")
            return None
    
    def get_comprehensive_sentiment(self, cryptocurrency):
        """Get comprehensive sentiment from all sources"""
        try:
            results = {}
            
            # Get Twitter sentiment
            twitter_sentiment = self.get_twitter_sentiment(cryptocurrency)
            if twitter_sentiment:
                results['twitter'] = twitter_sentiment
            
            # Get Reddit sentiment
            reddit_sentiment = self.get_reddit_sentiment(cryptocurrency)
            if reddit_sentiment:
                results['reddit'] = reddit_sentiment
            
            # Get news sentiment
            news_sentiment = self.get_news_sentiment(cryptocurrency)
            if news_sentiment:
                results['news'] = news_sentiment
            
            # Calculate overall sentiment
            if results:
                overall_sentiment = self.calculate_overall_sentiment(results)
                results['overall'] = overall_sentiment
            
            return results
            
        except Exception as e:
            print(f"Comprehensive sentiment analysis error: {str(e)}")
            return None
    
    def calculate_overall_sentiment(self, sentiment_data):
        """Calculate overall sentiment from multiple sources"""
        try:
            all_sentiments = []
            source_weights = {'twitter': 0.4, 'reddit': 0.3, 'news': 0.3}
            
            weighted_scores = {
                'textblob_polarity': 0,
                'vader_compound': 0,
                'total_weight': 0
            }
            
            for source, data in sentiment_data.items():
                if source in source_weights and 'aggregate_sentiment' in data:
                    weight = source_weights[source]
                    sentiment = data['aggregate_sentiment']
                    
                    weighted_scores['textblob_polarity'] += sentiment['textblob_polarity'] * weight
                    weighted_scores['vader_compound'] += sentiment['vader_compound'] * weight
                    weighted_scores['total_weight'] += weight
            
            if weighted_scores['total_weight'] > 0:
                final_sentiment = {
                    'textblob_polarity': weighted_scores['textblob_polarity'] / weighted_scores['total_weight'],
                    'vader_compound': weighted_scores['vader_compound'] / weighted_scores['total_weight']
                }
                
                # Classify sentiment
                if final_sentiment['vader_compound'] >= 0.05:
                    sentiment_label = 'Positive'
                elif final_sentiment['vader_compound'] <= -0.05:
                    sentiment_label = 'Negative'
                else:
                    sentiment_label = 'Neutral'
                
                final_sentiment['label'] = sentiment_label
                final_sentiment['confidence'] = abs(final_sentiment['vader_compound'])
                
                return final_sentiment
            
            return None
            
        except Exception as e:
            print(f"Overall sentiment calculation error: {str(e)}")
            return None
    
    def get_fear_greed_index(self):
        """Get Fear & Greed Index"""
        try:
            url = "https://api.alternative.me/fng/?limit=30"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return data['data']
            return None
            
        except Exception as e:
            print(f"Fear & Greed Index error: {str(e)}")
            return None
    
    def analyze_sentiment_trends(self, sentiment_history):
        """Analyze sentiment trends over time"""
        try:
            if not sentiment_history:
                return None
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(sentiment_history)
            
            # Calculate trend
            df['sentiment_score'] = df['vader_compound']
            trend = np.polyfit(range(len(df)), df['sentiment_score'], 1)[0]
            
            # Calculate volatility
            volatility = df['sentiment_score'].std()
            
            # Identify sentiment shifts
            sentiment_shifts = []
            for i in range(1, len(df)):
                if abs(df['sentiment_score'].iloc[i] - df['sentiment_score'].iloc[i-1]) > 0.3:
                    sentiment_shifts.append({
                        'date': df.index[i],
                        'shift': df['sentiment_score'].iloc[i] - df['sentiment_score'].iloc[i-1]
                    })
            
            return {
                'trend': 'Improving' if trend > 0 else 'Declining' if trend < 0 else 'Stable',
                'trend_strength': abs(trend),
                'volatility': volatility,
                'sentiment_shifts': sentiment_shifts,
                'current_sentiment': df['sentiment_score'].iloc[-1],
                'average_sentiment': df['sentiment_score'].mean()
            }
            
        except Exception as e:
            print(f"Sentiment trend analysis error: {str(e)}")
            return None
