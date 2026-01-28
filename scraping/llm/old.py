import asyncio
import json
import os
import re
from typing import List, Set
from urllib.parse import urljoin
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode, BrowserConfig
from bs4 import BeautifulSoup

BASE_URL = "http://www.gardenology.org"
MAIN_PAGE = f"{BASE_URL}/wiki/Main_Page"

async def get_category_links(crawler: AsyncWebCrawler) -> List[str]:
    """Extracts plant category links from the Main Page."""
    print(f"Fetching categories from {MAIN_PAGE}...")
    
    config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    result = await crawler.arun(url=MAIN_PAGE, config=config)
    
    if not result.success:
        print(f"Failed to crawl Main Page: {result.error_message}")
        return []

    soup = BeautifulSoup(result.html, 'html.parser')
    category_section = soup.find('span', id='Plant_Categories')
    if not category_section:
        print("Could not find 'Plant Categories' section.")
        return []

    # The table is a sibling of the h2 containing the span
    table = category_section.find_parent('h2').find_next_sibling('table')
    if not table:
        print("Could not find categories table.")
        return []

    links = []
    for a in table.find_all('a', href=True):
        href = a['href']
        if href.startswith('/wiki/'):
            full_url = urljoin(BASE_URL, href)
            if full_url not in links:
                links.append(full_url)
    
    print(f"Found {len(links)} category links.")
    return links

async def get_plant_links(crawler: AsyncWebCrawler, category_urls: List[str]) -> Set[str]:
    """Extracts individual plant links from category pages using arun_many."""
    print(f"Crawling {len(category_urls)} categories for plant links...")
    plant_links = set()
    
    # Use css_selector to target only the content area for cleaner extraction
    config = CrawlerRunConfig(
        cache_mode=CacheMode.ENABLED,
        css_selector="#mw-content-text"
    )
    
    # arun_many handles parallel crawling of all category pages
    results = await crawler.arun_many(urls=category_urls, config=config)
    
    async for result in results:
        if not result.success:
            continue
            
        # Use crawl4ai's built-in link extraction from the targeted section
        for link in result.links.get("internal", []):
            href = link.get("href", "")
            # Ignore redlinks (pages that don't exist yet)
            if 'redlink=1' in href or 'action=edit' in href:
                continue
            
            # Match /wiki/Plant_Name but avoid Special, Help, Category, etc.
            if href.startswith('/wiki/') and not any(x in href for x in [':', 'Main_Page', 'Copyright']):
                full_url = urljoin(BASE_URL, href)
                plant_links.add(full_url)
                
    print(f"Found {len(plant_links)} unique plant links.")
    return plant_links

async def main():
    # Use BrowserConfig for better control and performance
    browser_config = BrowserConfig(headless=True)
    
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # 1. Get Categories
        category_links = await get_category_links(crawler)
        if not category_links:
            return

        # 2. Get Plant Links from Categories
        all_plant_links = await get_plant_links(crawler, category_links)
        plant_list = list(all_plant_links)
        
        print(f"Checking relevance for {len(plant_list)} plants using arun_many...")
        
        relevant_plants = []
        config = CrawlerRunConfig(cache_mode=CacheMode.ENABLED)
        
        # Process all plants in parallel with streaming for real-time progress
        results = await crawler.arun_many(urls=plant_list, config=config)
        
        count = 0
        async for result in results:
            count += 1
            if result.success:
                # Check for "Pests and diseases" section in HTML
                if 'id="Pests_and_diseases"' in result.html or 'id="Pests_and_Diseases"' in result.html:
                    relevant_plants.append(result.url)
            
            if count % 100 == 0 or count == len(plant_list):
                print(f"Processed {count}/{len(plant_list)} plants. Found {len(relevant_plants)} so far...")

        # 4. Save results
        output_file = os.path.join("llm", "gardenology_relevant_urls.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(relevant_plants, f, indent=2)
        
        print(f"Finished! Found {len(relevant_plants)} relevant plants.")
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
