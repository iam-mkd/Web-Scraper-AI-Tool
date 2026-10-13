"""
Web Scraping & Anti-Bot Bypassing Package
"""
from app.scrapers.base_scraper import fetch_html, scrape_webpage
from app.scrapers.amazon_scraper import search_amazon_products, get_amazon_product_details

__all__ = [
    "fetch_html",
    "scrape_webpage",
    "search_amazon_products",
    "get_amazon_product_details",
]

