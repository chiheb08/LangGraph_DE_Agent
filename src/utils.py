import requests
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_medium_articles() -> list:
    api_key = os.getenv('MEDIUM_API_KEY')
    headers = {'Authorization': f'Bearer {api_key}'}
    url = 'https://api.medium.com/v1/articles?tag=data-engineering&limit=10'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        articles = response.json().get('data', [])
        return [article['content'] for article in articles]
    else:
        print(f"Failed to fetch articles: {response.status_code}")
        return [] 