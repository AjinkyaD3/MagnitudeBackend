from flask import Flask, request, jsonify
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import pandas as pd
import pickle
from datetime import datetime, date
from flask_cors import CORS


# Create Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


# Load the VADER model
try:
    with open('vader_model.pkl', 'rb') as f:
        sentiment_analyzer = pickle.load(f)
except FileNotFoundError:
    raise FileNotFoundError("Error: vader_model.pkl not found. Please run train_sentiment_model.py first.")
except Exception as e:
    raise Exception(f"Error loading model: {e}")

def fetch_stock_news(ticker):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    headers = {'User-Agent': 'Mozilla/5.0', 'Referer': 'https://finviz.com', 'Accept-Language': 'en-US,en;q=0.9'}
    url = finviz_url + ticker
    
    try:
        req = Request(url, headers=headers)
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')
        if not news_table:
            return None, "No news found for this ticker."
    except Exception as e:
        return None, f"Error fetching data: {e}"

    parsed_data = []
    today_str = date.today().strftime("%b-%d-%y")
    
    for row in news_table.find_all('tr'):
        title = row.a.text if row.a else 'No Title'
        date_data = row.td.text.strip().split()
        news_date_str = date_data[0] if len(date_data) == 2 else today_str
        news_time_str = date_data[-1]
        
        try:
            news_datetime = datetime.strptime(f"{news_date_str} {news_time_str}", "%b-%d-%y %I:%M%p")
            parsed_data.append([ticker, news_datetime, title])
        except:
            continue
    
    df = pd.DataFrame(parsed_data, columns=['ticker', 'datetime', 'title'])
    df = df.sort_values('datetime', ascending=False).head(10)
    df['compound'] = df['title'].apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
    
    return df

def get_prediction_text(average_sentiment):
    if average_sentiment >= 0.15:
        return {"prediction": "Stock Price Likely to Go Up", "sentiment": "positive"}
    elif average_sentiment <= -0.15:
        return {"prediction": "Stock Price Likely to Decline", "sentiment": "negative"}
    else:
        return {"prediction": "Stock Price Likely to Remain Stable", "sentiment": "neutral"}

@app.route('/analyze', methods=['GET'])
def analyze():
    ticker = request.args.get('ticker', '').strip().upper()
    if not ticker:
        return jsonify({"error": "Ticker symbol is required"}), 400
    
    result = fetch_stock_news(ticker)
    if not isinstance(result, pd.DataFrame):
        return jsonify({"error": result[1]}), 400
    
    df = result
    average_sentiment = df['compound'].mean()
    news_data = [
        {"datetime": row['datetime'].strftime("%Y-%m-%d %I:%M %p"), "title": row['title'], "sentiment_score": row['compound']} 
        for _, row in df.iterrows()
    ]
    
    return jsonify({
        "ticker": ticker,
        "news": news_data,
        "average_sentiment": average_sentiment,
        "prediction": get_prediction_text(average_sentiment)
    })

if __name__ == '__main__':
    app.run(debug=True)
