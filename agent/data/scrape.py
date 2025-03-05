import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
# import pdb

# List of URLs to scrape
urls = [
    "https://wise.com/help/articles/2452305/how-can-i-check-the-status-of-my-transfer",
    "https://wise.com/help/articles/2941900/when-will-my-money-arrive",
    "https://wise.com/help/articles/2977950/why-does-it-say-my-transfers-complete-when-the-money-hasnt-arrived-yet",
    "https://wise.com/help/articles/2977951/why-do-some-transfers-take-longer",
    "https://wise.com/help/articles/2932689/what-is-a-proof-of-payment",
    "https://wise.com/help/articles/2977938/whats-a-banking-partner-reference-number"
]

# User-Agent header to mimic a browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

def extract_article_content(url):
    try:
        # Add delay to avoid rate limiting
        time.sleep(1)
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract article title
        title_elem = soup.find('h1')
        title = title_elem.text.strip() if title_elem else "No title found"
        
        # Extract main content (this will need adjustment based on actual page structure)
        article_content = soup.find('article') or soup.find('div', class_='article-content')
        
        if article_content:
            # Extract all paragraphs
            for h4 in article_content.find_all('h4'):
                if 'Related articles' in h4.get_text(strip=True):
                    # Remove all elements after the "Related articles" section
                    for sibling in h4.find_all_next():
                        sibling.decompose()
                    break  # Stop after the first match
            paragraphs = article_content.find_all(['p', 'h2', 'h3', 'li'])
            content = "\n".join([p.text.strip() for p in paragraphs])
            
            # Find sublinks within the article
            sublinks = []
            for link in article_content.find_all('a', href=True):
                if '/help/articles/' in link['href']:
                    sublinks.append({
                        'text': link.text.strip(),
                        'url': link['href'] if link['href'].startswith('http') else f"https://wise.com{link['href']}"
                    })
            # pdb.set_trace()
            return {
                'url': url,
                'title': title,
                'content': content,
                'sublinks': sublinks
            }
        else:
            return {
                'url': url,
                'title': title,
                'content': "Could not extract content",
                'sublinks': []
            }
    except Exception as e:
        return {
            'url': url,
            'title': "Error",
            'content': f"Error: {str(e)}",
            'sublinks': []
        }
    

def recursive_scrape(start_urls, max_depth=2, visited=None):
    if visited is None:
        visited = set()
    
    all_data = []
    queue = [(url, 0) for url in start_urls]  # (url, depth)
    
    while queue:
        current_url, depth = queue.pop(0)
        
        if current_url in visited or depth > max_depth:
            continue
        
        visited.add(current_url)
        print(f"Scraping: {current_url}")
        
        article_data = extract_article_content(current_url)
        all_data.append(article_data)
        
        # Add sublinks to queue
        if depth < max_depth:
            for sublink in article_data['sublinks']:
                if sublink['url'] not in visited:
                    queue.append((sublink['url'], depth + 1))
    
    return all_data

# Execute scraping
faq_data = recursive_scrape(urls, max_depth=1)

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(faq_data)

# Save to CSV
df.to_csv('wise_faq_data1.csv', index=False)

# Create JSON format suitable for vector database
import json

# Format for vector database
vector_db_docs = []
for item in faq_data:
    doc = {
        'url': item['url'],
        'title': item['title'],
        'content': item['content'],
        'metadata': {
            'source': 'Wise FAQ',
            'section': 'Where is my money',
            'related_links': [sl['url'] for sl in item['sublinks']]
        }
    }
    vector_db_docs.append(doc)

# Save to JSON
with open('wise_faq_vector_db1.json', 'w') as f:
    json.dump(vector_db_docs, f, indent=2)