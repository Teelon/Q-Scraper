# Questrade Article Scraper

## Overview

This project consists of a web scraper that collects articles from the Questrade website. The system is designed to gather information about Questrade's investment services, and financial education resources for later use.

## Scraper Components

- `scraper.py`: The main web scraping script that collects articles from Questrade's website
- `scraper_clean.py`: A script for cleaning and processing the scraped data

## Data Storage

- `questrade_articles/`: Directory containing JSON files of scraped articles
- `questrade_articles_test/`: Test directory with sample article data

## Requirements

The project dependencies are listed in `requirements.txt`. Install them using:

```bash
pip install -r requirements.txt
```

## Data Structure

Each article is stored as a JSON file with the following structure:

- `title`: The article title
- `topics`: Array of related topics
- `key_points`: Array of key takeaways from the article
- `summary`: A brief summary of the article content
- `content`: The full article text
- `url`: The original URL source
- `scraped_at`: Timestamp of when the article was scraped

## Note

This project is for educational purposes only. The scraped content should not be used or construed as financial or investment advice.


## Example Output 

https://www.questrade.com/learning/questrade-basics/0-commissions-faq

![1745878028832](image/README/1745878028832.png)

## Last Updated

April 28, 2025
