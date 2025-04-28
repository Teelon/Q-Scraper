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
from typing import Dict, Any, List, Set, Tuple, Optional
import glob
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')  # Get API key from environment variable
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

MODEL = "gemini-1.5-flash"  # Using 1.5-flash as it's available
REQUESTS_PER_MINUTE = 25  # Rate limit for free tier
PAGE_TIMEOUT = 60000  # 60 seconds timeout for page operations
OUTPUT_DIR = "questrade_articles"  # Directory to save articles

class AIContentExtractor:
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.model = model
        genai.configure(api_key=api_key)
        self.gen_model = genai.GenerativeModel(model)
        self.request_times: List[float] = []

    async def extract_content(self, html_content: str, url: str) -> Dict[Any, Any]:
        """Extract structured content from HTML using AI"""
        # Manage rate limiting
        await self._manage_rate_limit()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title directly for reliability
        title_tag = soup.find('h1')
        title = title_tag.get_text().strip() if title_tag else "Unknown Title"
        
        # Find main content to reduce token usage
        main_content = soup.find('div', class_='content-blocks') or soup.find('main') or soup.find('article')
        
        # Prepare cleaned text for Gemini - more efficient than sending full HTML
        if main_content:
            # Extract just the text from paragraphs and headings in the main content
            content_elements = main_content.find_all(['p', 'h2', 'h3', 'h4', 'li'])
            cleaned_text = "\n\n".join([elem.get_text().strip() for elem in content_elements])
        else:
            # If no main content identified, extract text from the body
            body = soup.find('body')
            if body:
                # Remove scripts, styles, and hidden elements
                for script in body(["script", "style", "noscript", "iframe", "meta"]):
                    script.extract()
                cleaned_text = body.get_text()
                # Clean up whitespace
                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)
            else:
                cleaned_text = "Content extraction failed"
        
        # Truncate if too long
        if len(cleaned_text) > 100000:
            cleaned_text = cleaned_text[:100000] + "..."

        prompt = f"""
        Analyze this financial article from Questrade with title: "{title}"
        
        Extract and return ONLY this JSON structure:
        {{
            "title": "{title}",
            "topics": ["Topic 1", "Topic 2"],
            "key_points": ["Key point 1", "Key point 2"],
            "financial_data": {{
                "numbers": ["Any numerical data mentioned"],
                "metrics": ["Any financial metrics"]
            }},
            "summary": "A concise 2-3 sentence summary"
        }}
        
        Article text:
        {cleaned_text}
        """

        try:
            # Set generation config for more deterministic output
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            # Generate response
            response = self.gen_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            self.request_times.append(time.time())  # Track request time for rate limiting
            
            if not response.text:
                return self._fallback_extraction(soup, url, cleaned_text, title)

            # Parse and validate JSON response
            content = self._clean_json_response(response.text)
            
            # Add the full content, URL and timestamp
            content["content"] = cleaned_text
            content["url"] = url
            content["scraped_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            content["extraction_method"] = "ai"
            content["model"] = self.model
            
            return content
            
        except Exception as e:
            print(f"AI extraction failed: {str(e)}")
            return self._fallback_extraction(soup, url, cleaned_text, title)

    def _clean_json_response(self, response: str) -> Dict:
        """Clean and parse JSON from AI response"""
        content = response.strip()
        
        # Extract JSON if it's wrapped in code blocks
        json_match = re.search(r'```(?:json)?\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        
        # Clean up the content to ensure valid JSON
        content = content.strip()
        if content.startswith('```') and content.endswith('```'):
            content = content[3:-3].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            # If JSON parsing fails, create a minimal valid structure
            return {
                "title": "Error parsing response",
                "topics": [],
                "key_points": [],
                "financial_data": {"numbers": [], "metrics": []},
                "summary": "Failed to parse AI response."
            }

    def _fallback_extraction(self, soup: BeautifulSoup, url: str, cleaned_text: str = None, title: str = None) -> Dict:
        """Fallback extraction method when AI fails"""
        if not title:
            title_tag = soup.find('h1')
            title = title_tag.get_text().strip() if title_tag else "Untitled Article"
        
        # Use the already cleaned text if available
        if not cleaned_text:
            # Extract article content (main content div)
            content_div = soup.find('div', class_='content-blocks') or soup.find('main') or soup.find('article')
            
            # Extract article text
            article_content = ""
            if content_div:
                paragraphs = content_div.find_all(['p', 'h2', 'h3', 'h4', 'ul', 'ol'])
                for p in paragraphs:
                    article_content += p.get_text().strip() + "\n\n"
            
            cleaned_text = article_content.strip()
        
        # Extract metadata (if available)
        meta_description = soup.find('meta', {'name': 'description'})
        description = meta_description['content'] if meta_description and meta_description.has_attr('content') else ""
        
        # Create article object
        article = {
            "title": title,
            "content": cleaned_text,
            "topics": [],
            "key_points": [],
            "financial_data": {"numbers": [], "metrics": []},
            "summary": description,
            "url": url,
            "scraped_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "extraction_method": "fallback"
        }
        
        return article

    async def _manage_rate_limit(self):
        """Manage rate limiting for API requests"""
        # Keep only requests from the last minute
        current_time = time.time()
        recent_requests = [t for t in self.request_times if current_time - t < 60]
        
        # Update the request_times list
        self.request_times.clear()
        self.request_times.extend(recent_requests)
        
        # Check if we need to wait
        if len(recent_requests) >= REQUESTS_PER_MINUTE:
            # Calculate time to wait - until the oldest request is more than a minute old
            wait_time = 60 - (current_time - recent_requests[0])
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time + 1)  # Add 1 second buffer

class TimeEstimator:
    def __init__(self):
        self.start_time = time.time()
        self.processed_items = 0
        self.skipped_items = 0
        self.total_estimated_items = 0
        self.processing_times = []  # Store processing times for the last N items
        self.max_times_to_track = 10  # Track the last 10 items for moving average
    
    def set_total_items(self, total_items):
        """Set the total number of items to process for estimation"""
        self.total_estimated_items = total_items
    
    def item_processed(self, processing_time=None):
        """Record that an item was processed"""
        self.processed_items += 1
        if processing_time is not None:
            self.processing_times.append(processing_time)
            # Keep only the last N processing times
            if len(self.processing_times) > self.max_times_to_track:
                self.processing_times.pop(0)
    
    def item_skipped(self):
        """Record that an item was skipped (already processed)"""
        self.skipped_items += 1
    
    def get_average_processing_time(self):
        """Get the average processing time per item"""
        if not self.processing_times:
            # If no specific times recorded, use overall average
            elapsed = time.time() - self.start_time
            if self.processed_items == 0:
                return 0
            return elapsed / self.processed_items
        
        # Use the moving average of recent items
        return sum(self.processing_times) / len(self.processing_times)
    
    def get_elapsed_time(self):
        """Get the elapsed time in seconds"""
        return time.time() - self.start_time
    
    def get_estimated_remaining_time(self):
        """Get estimated remaining time in seconds"""
        avg_time = self.get_average_processing_time()
        remaining_items = self.total_estimated_items - (self.processed_items + self.skipped_items)
        
        # Adjust for skipped items - they're much faster
        if remaining_items <= 0:
            return 0
        
        # Estimate what percentage of remaining items might be skipped
        if self.processed_items + self.skipped_items > 0:
            skip_ratio = self.skipped_items / (self.processed_items + self.skipped_items)
            # We apply a conservative adjustment - skipped items take ~10% of normal processing time
            effective_remaining = remaining_items * (1 - skip_ratio * 0.9)
            return effective_remaining * avg_time
        
        return remaining_items * avg_time
    
    def get_time_stats(self):
        """Get formatted time statistics"""
        elapsed = self.get_elapsed_time()
        elapsed_str = format_time_duration(elapsed)
        
        remaining = self.get_estimated_remaining_time()
        remaining_str = format_time_duration(remaining)
        
        if self.total_estimated_items > 0:
            progress_pct = 100 * (self.processed_items + self.skipped_items) / self.total_estimated_items
        else:
            progress_pct = 0
        
        # Calculate estimated completion time
        now = datetime.now()
        completion_time = now + timedelta(seconds=remaining)
        completion_str = completion_time.strftime("%H:%M:%S")
        
        # Build progress bar
        progress_bar_length = 30
        filled_length = int(progress_bar_length * (self.processed_items + self.skipped_items) / max(1, self.total_estimated_items))
        bar = '█' * filled_length + '░' * (progress_bar_length - filled_length)
        
        return {
            "elapsed": elapsed_str,
            "remaining": remaining_str, 
            "progress_pct": progress_pct,
            "completion_time": completion_str,
            "progress_bar": bar,
            "avg_time_per_item": self.get_average_processing_time()
        }

def format_time_duration(seconds):
    """Format time duration in seconds to a human-readable string"""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

async def get_pagination_info(page, url) -> Tuple[int, int]:
    """Extract total number of articles and pages"""
    try:
        # Navigate to the page
        await page.goto(url, timeout=PAGE_TIMEOUT)
        
        # Wait for content to load, but don't use networkidle which can time out
        await asyncio.sleep(5)  # Simple delay instead of networkidle
        
        # Extract total articles count
        html_content = await page.content()
        soup = BeautifulSoup(html_content, 'html.parser')
        results_text = None
        
        # Try different methods to find the results count
        for text in soup.stripped_strings:
            match = re.search(r"Results (\d+) - (\d+) of about (\d+)", text)
            if match:
                results_text = text
                break
        
        if results_text:
            match = re.search(r"of about (\d+)", results_text)
            total_articles = int(match.group(1)) if match else 500
        else:
            # Default if we can't find the count
            print("Could not find pagination info, using default values")
            total_articles = 500
        
        # Calculate total pages (20 results per page)
        articles_per_page = 20
        total_pages = (total_articles + articles_per_page - 1) // articles_per_page
        
        return total_articles, total_pages
        
    except Exception as e:
        print(f"Error getting pagination info: {str(e)}")
        # Return reasonable defaults
        return 500, 25

async def extract_article_links(page) -> List[str]:
    """Extract article links from the current search results page"""
    # Wait a moment for JavaScript to finish executing
    await asyncio.sleep(3)
    
    html_content = await page.content()
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all search result items
    article_links = []
    
    # Try different selectors that might contain article links
    result_items = soup.find_all('h3')
    
    for item in result_items:
        link_tag = item.find('a')
        if link_tag and link_tag.has_attr('href'):
            article_url = urljoin("https://www.questrade.com", link_tag['href'])
            article_links.append(article_url)
    
    # If we didn't find any links with h3 tags, try another approach
    if not article_links:
        all_links = soup.find_all('a')
        for link in all_links:
            href = link.get('href', '')
            # Look for links that might be articles
            if '/learning/' in href and not '/learning/search' in href:
                article_url = urljoin("https://www.questrade.com", href)
                article_links.append(article_url)
    
    # Remove duplicates while preserving order
    unique_links = []
    seen = set()
    for link in article_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
    
    return unique_links

def create_safe_filename(url):
    """Create a safe filename from URL"""
    # Extract the last part of the URL
    base = url.rstrip('/').split('/')[-1]
    
    # Remove query parameters if any
    base = base.split('?')[0]
    
    # Replace special characters
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base)
    
    # Add json extension
    return f"{safe_name}.json"

def get_existing_files() -> Dict[str, str]:
    """Get a dictionary of existing files with URL to filename mapping"""
    existing_files = {}
    
    # Ensure the output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        return existing_files
    
    # Get all JSON files in the output directory
    file_paths = glob.glob(os.path.join(OUTPUT_DIR, "*.json"))
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # If the file contains a URL field, add it to our mapping
                if 'url' in data:
                    url = data['url']
                    existing_files[url] = os.path.basename(file_path)
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
    
    return existing_files

def should_update_file(file_path: str, update_threshold_days: int = 30) -> bool:
    """Check if a file should be updated based on its age"""
    if not os.path.exists(file_path):
        return True
    
    file_mod_time = os.path.getmtime(file_path)
    current_time = time.time()
    days_since_update = (current_time - file_mod_time) / (60 * 60 * 24)
    
    return days_since_update > update_threshold_days

def print_progress_update(time_estimator, stats, current_page, total_pages):
    """Print a progress update with time estimates"""
    time_stats = time_estimator.get_time_stats()
    
    # Clear the terminal
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # Print progress bar and statistics
    print(f"\n{'=' * 80}")
    print(f"QUESTRADE ARTICLE SCRAPER - PROGRESS UPDATE")
    print(f"{'=' * 80}")
    
    print(f"\nProgress: {time_stats['progress_bar']} {time_stats['progress_pct']:.1f}%")
    print(f"Page {current_page}/{total_pages}")
    
    print(f"\nTime Statistics:")
    print(f"  Elapsed Time:  {time_stats['elapsed']}")
    print(f"  Remaining:     {time_stats['remaining']}")
    print(f"  ETA:           {time_stats['completion_time']}")
    
    print(f"\nArticle Statistics:")
    print(f"  Total Processed: {stats['total_processed']}")
    print(f"  New Articles:    {stats['new_articles']} ")
    print(f"  Skipped:         {stats['skipped_existing']} (already up-to-date)")
    print(f"  Updated:         {stats['updated_articles']} (older articles refreshed)")
    print(f"  Errors:          {stats['errors']}")
    
    print(f"\nAverage processing time per article: {format_time_duration(time_stats['avg_time_per_item'])}")
    print(f"{'=' * 80}\n")

async def main():
    # Print startup message
    print("Starting Questrade article scraper...")
    print(f"Using model: {MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Create directory for saving articles if it doesn't exist
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    
    # Get existing files information
    existing_files = get_existing_files()
    print(f"Found {len(existing_files)} existing files in the output directory")
    
    # Initialize the content extractor
    print("Initializing AI content extractor...")
    extractor = AIContentExtractor(api_key=GEMINI_API_KEY, model=MODEL)
    
    # Initialize time estimator
    time_estimator = TimeEstimator()
    
    # Base URL and search parameters
    base_url = "https://www.questrade.com/learning/search"
    search_params = "?indexCatalogue=learnhub-indexing&searchQuery=a&wordsMode=AllWords"
    
    # Track articles to avoid duplicates within this run
    processed_articles: Set[str] = set()
    
    # Set update threshold (default: 30 days)
    update_threshold_days = 30
    
    # Stats tracking
    stats = {
        "total_processed": 0,
        "new_articles": 0,
        "skipped_existing": 0,
        "updated_articles": 0,
        "errors": 0
    }
    
    try:
        async with async_playwright() as p:
            # Launch browser with additional options for stability
            browser = await p.chromium.launch(
                headless=False,  # Set to True for production
                args=['--disable-web-security', '--disable-features=IsolateOrigins,site-per-process']
            )
            
            # Create a context with more generous timeouts
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            # Create page with timeout setting
            page = await context.new_page()
            page.set_default_timeout(PAGE_TIMEOUT)
            
            print(f"Navigating to search page: {base_url}{search_params}")
            
            # Determine total number of pages and articles
            total_articles, total_pages = await get_pagination_info(page, f"{base_url}{search_params}")
            print(f"Found approximately {total_articles} articles across {total_pages} pages")
            
            # Set the total items for the time estimator
            time_estimator.set_total_items(total_articles)
            
            # For testing or limiting, you can set a max number of pages
            max_pages_to_process = total_pages  # Process all pages
            # max_pages_to_process = 2  # Uncomment to limit to first 2 pages for testing
            
            # Update progress every 15 seconds
            last_progress_update = time.time()
            update_interval = 15  # seconds
            
            # Iterate through pages
            for page_num in range(1, min(total_pages + 1, max_pages_to_process + 1)):
                if page_num == 1:
                    current_url = f"{base_url}{search_params}"
                else:
                    current_url = f"{base_url}/{page_num}{search_params}"
                
                print(f"Processing page {page_num}/{total_pages}: {current_url}")
                
                try:
                    # Navigate to the page
                    await page.goto(current_url, timeout=PAGE_TIMEOUT)
                    # Simple delay instead of waiting for networkidle
                    await asyncio.sleep(5)
                    
                    # Extract article links from current page
                    article_links = await extract_article_links(page)
                    print(f"Found {len(article_links)} articles on page {page_num}")
                    
                    # For testing or limiting, you can set a max number of articles per page
                    max_articles_to_process = len(article_links)  # Process all articles
                    # max_articles_to_process = 3  # Uncomment to limit to first 3 articles for testing
                    
                    # Process each article
                    for i, article_url in enumerate(article_links[:max_articles_to_process]):
                        stats["total_processed"] += 1
                        
                        # Skip if we've already processed this article in this run
                        if article_url in processed_articles:
                            print(f"Skipping already processed article in this run: {article_url}")
                            continue
                            
                        processed_articles.add(article_url)
                        
                        # Create a safe filename from the URL
                        filename = create_safe_filename(article_url)
                        output_path = os.path.join(OUTPUT_DIR, filename)
                        
                        # Check if we already have this article in our existing files
                        if article_url in existing_files:
                            existing_file_path = os.path.join(OUTPUT_DIR, existing_files[article_url])
                            
                            # Check if the file needs updating based on age
                            if not should_update_file(existing_file_path, update_threshold_days):
                                print(f"Skipping article (less than {update_threshold_days} days old): {article_url}")
                                stats["skipped_existing"] += 1
                                time_estimator.item_skipped()
                                
                                # Update progress display if it's time
                                current_time = time.time()
                                if current_time - last_progress_update > update_interval:
                                    print_progress_update(time_estimator, stats, page_num, total_pages)
                                    last_progress_update = current_time
                                
                                continue
                            else:
                                print(f"Updating article (older than {update_threshold_days} days): {article_url}")
                                stats["updated_articles"] += 1
                        else:
                            print(f"Processing new article {i+1}/{len(article_links)}: {article_url}")
                            stats["new_articles"] += 1
                        
                        # Record start time for this article
                        article_start_time = time.time()
                        
                        try:
                            # Navigate to the article page
                            await page.goto(article_url, timeout=PAGE_TIMEOUT)
                            # Simple delay instead of waiting for networkidle
                            await asyncio.sleep(5)
                            
                            # Extract article HTML
                            article_html = await page.content()
                            
                            # Use Gemini to extract structured data
                            article_content = await extractor.extract_content(article_html, article_url)
                            
                            # Save article to file
                            with open(output_path, 'w', encoding='utf-8') as f:
                                json.dump(article_content, f, indent=2)
                                
                            print(f"Saved article to {output_path}")
                            
                            # Calculate processing time and update estimator
                            processing_time = time.time() - article_start_time
                            time_estimator.item_processed(processing_time)
                            
                        except Exception as e:
                            print(f"Error processing article {article_url}: {str(e)}")
                            stats["errors"] += 1
                            time_estimator.item_processed()  # Still count as processed for time estimation
                        
                        # Update progress display if it's time
                        current_time = time.time()
                        if current_time - last_progress_update > update_interval:
                            print_progress_update(time_estimator, stats, page_num, total_pages)
                            last_progress_update = current_time
                    
                except Exception as e:
                    print(f"Error processing page {current_url}: {str(e)}")
                    stats["errors"] += 1
                
                # Add delay between pages to be respectful to the server
                await asyncio.sleep(3)
                
                # Update progress at the end of each page
                print_progress_update(time_estimator, stats, page_num, total_pages)
                last_progress_update = time.time()
            
            # Close the browser
            await browser.close()
            
            # Print final statistics
            print("\n=== Final Scraping Statistics ===")
            print(f"Total articles processed: {stats['total_processed']}")
            print(f"New articles added: {stats['new_articles']}")
            print(f"Existing articles skipped: {stats['skipped_existing']}")
            print(f"Older articles updated: {stats['updated_articles']}")
            print(f"Errors encountered: {stats['errors']}")
            print(f"Total articles in database: {len(get_existing_files())}")
            print("=========================\n")
            
            print(f"Scraping complete. Processed {len(processed_articles)} articles.")
            print(f"Total elapsed time: {format_time_duration(time_estimator.get_elapsed_time())}")
    
    except Exception as e:
        print(f"Critical error: {str(e)}")

if __name__ == "__main__":
    # Run the main function
    print("Starting scraper...")
    asyncio.run(main())
    print("Scraper finished.")