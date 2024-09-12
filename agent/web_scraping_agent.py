import requests
from bs4 import BeautifulSoup

class WebScraper:
    def scrape(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            text = ' '.join([p.text for p in soup.find_all('p')])
            return text
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
            return None
