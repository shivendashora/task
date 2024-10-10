from flask import Flask, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
import feedparser
import joblib
from datetime import datetime
import pymysql
pymysql.install_as_MySQLdb()


# Initialize Flask app
app = Flask(__name__)

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/articles'
db = SQLAlchemy(app)

# Load the SVM model and vectorizer
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Define the Article model
class articles(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String, unique=True, nullable=False)
    content = db.Column(db.Text, nullable=False)
    link = db.Column(db.String, nullable=False)
    Date = db.Column(db.String, nullable=False)
    Category = db.Column(db.String, nullable=False)

    def __repr__(self):
        return f'<Article {self.title}>'

# Create the database tables
with app.app_context():
    db.create_all()

# List of RSS Feeds
rss_feeds = [
    'http://rss.cnn.com/rss/cnn_topstories.rss',
    'http://qz.com/feed',
    'http://feeds.foxnews.com/foxnews/politics',
    'http://feeds.reuters.com/reuters/businessNews',
    'http://feeds.feedburner.com/NewshourWorld',
    'https://feeds.bbci.co.uk/news/world/asia/india/rss.xml'
]

def fetch_articles():
    articles = []
    
    for feed in rss_feeds:
        feed_data = feedparser.parse(feed)
        
        for entry in feed_data.entries:
            # Get the title, content (if available), link, and published date
            title = entry.title
            link = entry.link
            content = entry.summary if 'summary' in entry else 'No content available'
            
            # Get the publication date, convert it to a readable format
            published_date = entry.published if 'published' in entry else 'No date available'
            try:
                published_date = datetime(*entry.published_parsed[:6]).strftime('%Y-%m-%d %H:%M:%S') if 'published_parsed' in entry else 'No date available'
            except Exception as e:
                published_date = 'No date available'
            
            # Classify the article based on the title
            category = classify_article(title)
            
            # Beautify content (for example, keeping a minimum of 3 paragraphs)
            content = beautify_content(content)
            
            # Append article to the list
            # In fetch_articles(), when appending to the list:
            articles.append({
            'Name': title,                      # Changed 'title' to 'name'
            'Content': content,
            'Link': link,
            'Date': published_date,   # Keep this as is, represents the "Date" column
            'Category': category
            })

    
    return articles

# Beautify content (ensure at least 3 paragraphs)
def beautify_content(content):
    paragraphs = content.split('\n\n')
    if len(paragraphs) < 3:
        # Add dummy paragraphs if less than 3
        paragraphs += [''] * (3 - len(paragraphs))
    return '\n\n'.join(paragraphs[:3])

# Classify article based on its title
def classify_article(title):
    title_vectorized = vectorizer.transform([title])
    prediction = svm_model.predict(title_vectorized)
    return prediction[0]

# Route to serve the HTML page and fetch/store articles
@app.route('/')
def index():
    # Fetch articles and store them in the database
    fetched_articles = fetch_articles()
    for article in fetched_articles:
        try:
            new_article = articles(
                Name=article['Name'],
                content=article['Content'],
                link=article['Link'],
                Date=article['Date'],
                Category=article['Category']
            )
            db.session.add(new_article)
            db.session.commit()
        except Exception as e:
            print(f"Article '{article['Name']}' already exists in the database or an error occurred: {e}")

    # Query all stored articles from the database
    stored_articles = articles.query.all()

    # Pass the articles to the template
    return render_template('index.html', articles=stored_articles)


# Run the application
if __name__ == '__main__':
    app.run(debug=True)
