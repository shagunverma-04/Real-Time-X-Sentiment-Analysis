import os
import re
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, render_template, request, flash, jsonify
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TweetSentimentAnalyzer:
    def __init__(self, model_path='models/advanced_sentiment_model.h5', dataset_path='twitter_training.csv'):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.model = None
        self.tokenizer = None
        self.label_encoder = LabelEncoder()
        self.max_length = 100
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Initialize model
        self.load_or_train_model()

    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove mentions
        text = re.sub(r'@\w+', '', text)
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        # Remove special characters but keep emojis
        text = re.sub(r'[^\w\s\u263a-\U0001f645]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def load_or_train_model(self):
        """Load existing model or train new one with error handling"""
        try:
            if os.path.exists(self.model_path):
                print("Loading existing model...")
                self.model = tf.keras.models.load_model(self.model_path)
                print("Model loaded successfully")
                return
            print("No existing model found. Training new model...")
            self.train_comprehensive_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Training new model...")
            self.train_comprehensive_model()

    def train_comprehensive_model(self):
        """Train model with fixed input size and improved architecture"""
        try:
            # Load and prepare data
            df = pd.read_csv(self.dataset_path, encoding='utf-8', on_bad_lines='skip')
            df.columns = ["ID", "Category", "Sentiment", "Tweet"]
            df = df[["Sentiment", "Tweet"]].dropna()
            
            # Preprocess texts
            texts = df['Tweet'].apply(self.preprocess_text).tolist()
            labels = df['Sentiment'].tolist()
            
            # Encode labels
            self.label_encoder.fit(labels)
            encoded_labels = self.label_encoder.transform(labels)
            categorical_labels = tf.keras.utils.to_categorical(encoded_labels)
            
            # Prepare text sequences
            self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
                num_words=5000, 
                oov_token="<OOV>"
            )
            self.tokenizer.fit_on_texts(texts)
            sequences = self.tokenizer.texts_to_sequences(texts)
            padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
                sequences, 
                maxlen=self.max_length,
                padding='post',
                truncating='post'
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                padded_sequences,
                categorical_labels,
                test_size=0.2,
                random_state=42
            )
            
            # Build model with fixed input shape
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.max_length,), dtype=tf.int32),
                tf.keras.layers.Embedding(5000, 128, input_length=self.max_length),
                tf.keras.layers.SpatialDropout1D(0.2),
                tf.keras.layers.LSTM(64, return_sequences=True),
                tf.keras.layers.LSTM(32),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(len(self.label_encoder.classes_), activation='softmax')
            ])

            # Use mixed precision training for better performance 
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            # Compile and train
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model.fit(
                X_train,
                y_train,
                epochs=10,
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=1
            )
            
            # Save model
            self.model.save(self.model_path)
            print("Model trained and saved successfully")
            
        except Exception as e:
            print(f"Error during model training: {e}")
            raise

    def predict_sentiment(self, text):
        """Predict sentiment with error handling"""
        try:
            if not self.model or not self.tokenizer:
                print("Model or tokenizer not loaded.")
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            processed_text = self.preprocess_text(text)
            if not processed_text:
                return {'sentiment': 'neutral', 'confidence': 0.0}
            
            sequence = self.tokenizer.texts_to_sequences([processed_text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(
                sequence, maxlen=self.max_length, padding='post', truncating='post'
            )
            
            prediction = self.model.predict(padded_sequence, verbose=0)[0]
            predicted_class = self.label_encoder.classes_[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            return {'sentiment': predicted_class, 'confidence': confidence}
        
        except Exception as e:
            print(f"Error in sentiment prediction: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}

def fetch_tweets(query, limit=10, retries=3):
    """Enhanced tweet fetching with authentication handling"""
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-notifications")
    # Add these new options to better handle modern Twitter
    options.add_argument("--enable-javascript")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--lang=en")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    tweets = []
    
    print(f"Starting tweet fetch for query: {query}")

    for attempt in range(retries):
        driver = None
        try:
            print(f"Attempt {attempt + 1} of {retries}")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            # Update the stealth settings
            driver.execute_cdp_cmd('Network.setUserAgentOverride', {
                "userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            })
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            # Try Nitter as an alternative to Twitter
            nitter_instances = [
                "https://nitter.net",
                "https://nitter.lacontrevoie.fr",
                "https://nitter.1d4.us"
            ]
            
            for instance in nitter_instances:
                try:
                    search_url = f"{instance}/search?f=tweets&q={query}"
                    print(f"Trying Nitter instance: {search_url}")
                    
                    driver.get(search_url)
                    time.sleep(5)  # Wait for page load
                    
                    # Nitter specific selectors
                    tweet_elements = driver.find_elements(By.CSS_SELECTOR, ".timeline-item")
                    
                    if tweet_elements:
                        print(f"Found {len(tweet_elements)} tweets on {instance}")
                        
                        for tweet in tweet_elements[:limit]:
                            try:
                                tweet_text = tweet.find_element(By.CSS_SELECTOR, ".tweet-content").text
                                if tweet_text and tweet_text not in [t['text'] for t in tweets]:
                                    tweets.append({'text': tweet_text})
                                    print(f"Found tweet: {tweet_text[:50]}...")
                            except Exception as e:
                                print(f"Error extracting tweet text: {str(e)}")
                                continue
                        
                        if tweets:
                            break  # Found tweets, exit instance loop
                            
                except Exception as e:
                    print(f"Error with Nitter instance {instance}: {str(e)}")
                    continue
            
            if tweets:
                break  # Exit retry loop if we have tweets
                
        except Exception as e:
            print(f"Error during tweet fetching: {str(e)}")
        
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            print(f"Attempt {attempt + 1} completed")
    
    if not tweets:
        print("Failed to fetch any tweets after all attempts")
    
    return tweets[:limit]

# Flask application setup
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize sentiment analyzer
sentiment_analyzer = TweetSentimentAnalyzer()

def calculate_sentiment_stats(analyzed_tweets):
    """
    Calculate sentiment statistics from analyzed tweets
    
    Args:
        analyzed_tweets (list): List of dictionaries containing analyzed tweets
        
    Returns:
        dict: Dictionary containing percentages for each sentiment category
    """
    try:
        # Get list of all sentiments
        ml_sentiments = [t['ml_sentiment'] for t in analyzed_tweets]
        total = len(ml_sentiments)
        
        if total == 0:
            return {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
        
        # Calculate percentages for each sentiment
        stats = {
            'positive': round(ml_sentiments.count('positive') / total * 100, 1),
            'negative': round(ml_sentiments.count('negative') / total * 100, 1),
            'neutral': round(ml_sentiments.count('neutral') / total * 100, 1)
        }
        
        return stats
        
    except Exception as e:
        print(f"Error calculating sentiment stats: {str(e)}")
        return {
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

@app.route('/sentiment', methods=['GET', 'POST'])
def sentiment_analysis():
    if request.method == 'GET':
        return render_template('sentiment.html')
    
    try:
        userid = request.form.get('userid', '').strip()
        hashtag = request.form.get('hashtag', '').strip()
        
        if not userid and not hashtag:
            flash("Please enter either a user ID or hashtag", "error")
            return render_template('sentiment.html')
        
        if userid and hashtag:
            flash("Please enter only one: user ID or hashtag", "error")
            return render_template('sentiment.html')
        
        # Set a higher limit for tweet fetching (e.g., 20)
        query = f"from:{userid}" if userid else f"#{hashtag}"
        tweets = fetch_tweets(query, limit=20)  # Increased limit
        
        if not tweets:
            flash("No tweets found. Please check the user ID/hashtag and try again.", "error")
            return render_template('sentiment.html')
        
        print(f"Found {len(tweets)} tweets to analyze")  # Debugging
        
        # Process all tweets
        analyzed_tweets = []
        for tweet in tweets:
            try:
                # Print the tweet being analyzed for debugging
                print(f"Analyzing tweet: {tweet['text'][:100]}...")
                
                ml_sentiment = sentiment_analyzer.predict_sentiment(tweet['text'])
                analyzed_tweet = {
                    'text': tweet['text'],
                    'ml_sentiment': ml_sentiment['sentiment'],
                    'ml_confidence': round(float(ml_sentiment['confidence']), 3)
                }
                analyzed_tweets.append(analyzed_tweet)
                print(f"Analysis complete - Sentiment: {ml_sentiment['sentiment']}")  # Debugging
                
            except Exception as e:
                print(f"Error analyzing tweet: {str(e)}")
                continue
        
        if not analyzed_tweets:
            flash("Could not analyze any tweets. Please try again.", "error")
            return render_template('sentiment.html')
        
        print(f"Successfully analyzed {len(analyzed_tweets)} tweets")  # Debugging
        
        # Calculate statistics
        stats = calculate_sentiment_stats(analyzed_tweets)
        
        return render_template(
            'sentiment.html',
            positive=stats['positive'],
            negative=stats['negative'],
            neutral=stats['neutral'],
            analyzed_tweets=analyzed_tweets  # Pass all analyzed tweets to template
        )
        
    except Exception as e:
        print(f"Error in sentiment analysis route: {str(e)}")  # Debugging
        flash(f"An error occurred: {str(e)}", "error")
        return render_template('sentiment.html')

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)