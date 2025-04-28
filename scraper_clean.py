import os
import time
import re
import json
import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Get API key from environment variable
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

MODEL = "gemini-1.5-flash"  # Using 1.5-flash model
OUTPUT_DIR = "questrade_articles_test"  # Directory to save articles

class AIContentExtractor:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(MODEL)
        
    async def extract_content(self, html_content, url):
        """Extract structured content from HTML using AI"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title_tag = soup.find('h1')
        title = title_tag.get_text().strip() if title_tag else "Unknown Title"
        
        # Find main content
        main_content = soup.find('div', class_='content-blocks') or soup.find('main') or soup.find('article')
        
        # Extract content text
        if (main_content):
            content_elements = main_content.find_all(['p', 'h2', 'h3', 'h4', 'li'])
            cleaned_text = "\n\n".join([elem.get_text().strip() for elem in content_elements])
        else:
            # If no main content identified, extract text from the body
            body = soup.find('body')
            if body:
                for script in body(["script", "style", "noscript", "iframe", "meta"]):
                    script.extract()
                cleaned_text = body.get_text()
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
            else:
                cleaned_text = "Content extraction failed"

        # Prompt for Gemini
        prompt = f"""
        Analyze this financial article from Questrade with title: "{title}"
        
        Extract and return ONLY this JSON structure:
        {{
            "title": "{title}",
            "topics": ["Topic 1", "Topic 2"],
            "key_points": ["Key point 1", "Key point 2"],
            "summary": "A concise 2-3 sentence summary"
        }}
        
        Article text:
        {cleaned_text}
        """

        try:
            # Generate response
            response = self.gen_model.generate_content(prompt)
            
            # Parse JSON response
            content = self._parse_json(response.text)
            
            # Add the full content and URL
            content["content"] = cleaned_text
            content["url"] = url
            content["scraped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return content
            
        except Exception as e:
            print(f"AI extraction failed: {str(e)}")
            # Return basic structure if AI fails
            return {
                "title": title,
                "content": cleaned_text,
                "url": url,
                "topics": [],
                "key_points": [],
                "summary": "Extraction failed",
                "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def _parse_json(self, text):
        """Parse JSON from AI response"""
        # Extract JSON if it's wrapped in code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1)
        
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Return a basic structure if parsing fails
            return {
                "title": "Error parsing response",
                "topics": [],
                "key_points": [],
                "summary": "Failed to parse AI response."
            }

def create_filename(url):
    """Create a safe filename from URL"""
    base = url.rstrip('/').split('/')[-1]
    base = base.split('?')[0]
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base)
    return f"{safe_name}.json"

async def extract_article_links(page):
    """Extract article links from the current search results page"""
    html_content = await page.content()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    article_links = []
    result_items = soup.find_all('h3')
    
    for item in result_items:
        link_tag = item.find('a')
        if link_tag and link_tag.has_attr('href'):
            article_url = urljoin("https://www.questrade.com", link_tag['href'])
            article_links.append(article_url)
    
    # Remove duplicates
    unique_links = []
    seen = set()
    for link in article_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    
    return unique_links

async def main():
    print("Starting Questrade article scraper...")
    
    # Create directory for saving articles if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Initialize the content extractor
    extractor = AIContentExtractor(api_key=GEMINI_API_KEY)
    
    # Base URL for search
    search_url = "https://www.questrade.com/learning/search?indexCatalogue=learnhub-indexing&searchQuery=a&wordsMode=AllWords"
    
    # Track processed articles
    processed_articles = set()
    
    # Stats tracking
    stats = {
        "processed": 0,
        "saved": 0,
        "errors": 0
    }
    
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=False)  # Set to True for production
            context = await browser.new_context(viewport={'width': 1280, 'height': 800})
            page = await context.new_page()
            
            print(f"Navigating to search page: {search_url}")
            await page.goto(search_url, timeout=60000)
            await asyncio.sleep(3)
            
            # For demo purposes, we'll just process the first page
            # Extract article links
            article_links = await extract_article_links(page)
            print(f"Found {len(article_links)} articles on page")
            
            # Limit to first 5 articles for the demo
            max_articles = min(5, len(article_links))
            print(f"Processing first {max_articles} articles")
            
            # Process each article
            for i, article_url in enumerate(article_links[:max_articles]):
                stats["processed"] += 1
                processed_articles.add(article_url)
                
                print(f"Processing article {i+1}/{max_articles}: {article_url}")
                
                try:
                    # Navigate to the article page
                    await page.goto(article_url, timeout=60000)
                    await asyncio.sleep(3)
                    
                    # Extract article HTML
                    article_html = await page.content()
                    
                    # Use Gemini to extract structured data
                    article_content = await extractor.extract_content(article_html, article_url)
                    
                    # Save article to file
                    filename = create_filename(article_url)
                    output_path = os.path.join(OUTPUT_DIR, filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(article_content, f, indent=2)
                        
                    print(f"Saved article to {output_path}")
                    stats["saved"] += 1
                    
                except Exception as e:
                    print(f"Error processing article: {str(e)}")
                    stats["errors"] += 1
                
                # Add delay between articles
                await asyncio.sleep(1)
            
            # Close the browser
            await browser.close()
            
            # Print final statistics
            print("\n=== Scraping Complete ===")
            print(f"Articles processed: {stats['processed']}")
            print(f"Articles saved: {stats['saved']}")
            print(f"Errors encountered: {stats['errors']}")
            print("=======================")
    
    except Exception as e:
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())