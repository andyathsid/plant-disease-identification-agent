import json
import os
import asyncio
import logging
import sys
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import List, Literal, Optional
from urllib.parse import urljoin
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph
from crawl4ai import AsyncWebCrawler

# Load environment variables (ensure GOOGLE_API_KEY is set in your .env)
load_dotenv()

# Silence ScrapeGraphAI and other libraries
logging.getLogger("scrapegraphai").setLevel(logging.WARNING)
logging.getLogger("crawl4ai").setLevel(logging.WARNING)

# 1. Define the schema based on structure.json
class PestDisease(BaseModel):
    category: Literal["fungal", "bacterial", "viral", "pest"]
    name: str
    symptoms: str
    causes: str
    management: str

class PlantInfo(BaseModel):
    plant: str
    description: str
    propagation: str
    pests_and_diseases: List[PestDisease]

# 2. Configure the graph
graph_config = {
    "llm": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "model": "google_genai/gemini-2.5-flash",
        "temperature": 0.0,
    },
    "verbose": True,
    "headless": True,
}

async def get_plant_links():
    """Extracts the plant info URLs from the main directory using crawl4ai."""
    print("Fetching plant links from directory using crawl4ai...")
    
    base_url = "https://plantvillage.psu.edu"
    directory_url = f"{base_url}/plants"
    
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=directory_url)
        
        if not result.success:
            print(f"Failed to crawl directory: {result.error_message}")
            return []

        # Extract links from the crawl result
        # Crawl4AI provides internal links in the result.links["internal"]
        links = []
        for link in result.links.get("internal", []):
            href = link.get("href", "")
            # Filter for /topics/[plant-name] and append /infos if not present
            if "/topics/" in href and not href.endswith("/infos"):
                full_url = urljoin(base_url, href)
                if not full_url.endswith("/infos"):
                    full_url = full_url.rstrip("/") + "/infos"
                if full_url not in links:
                    links.append(full_url)
            elif href.endswith("/infos"):
                full_url = urljoin(base_url, href)
                if full_url not in links:
                    links.append(full_url)
        
        print(f"Found {len(links)} potential plant info links.")
        return links

def scrape_plant_details(url: str):
    """Scrapes a single plant info page using the defined schema."""
    print(f"Scraping details for: {url}")
    
    prompt = "Extract the plant name, description, propagation details, and a list of pests and diseases with their category, symptoms, causes, and management. If information is missing, use empty strings or empty lists."
    
    smart_scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=graph_config,
        schema=PlantInfo # Enforce your structure.json
    )
    
    # Use redirect_stdout/stderr to catch any internal prints from the library
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            return smart_scraper.run()

async def main():
    # 1. Discover links
    all_links = await get_plant_links()
    
    if not all_links:
        print("No links found. Exiting.")
        return

    output_path = os.path.join("llm", "data", "plantvillage.jsonl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Starting extraction for {len(all_links)} plants...")
    
    # 2. Process one by one for memory efficiency
    for i, url in enumerate(all_links):
        try:
            # We wrap the sync scrapegraph call in a thread to keep the loop responsive if needed,
            # but for a simple script, running it directly is fine as long as we don't block asyncio.
            # SmartScraperGraph.run() is sync.
            data = await asyncio.to_thread(scrape_plant_details, url)
            
            # Handle Pydantic model serialization
            if isinstance(data, BaseModel):
                data_dict = data.model_dump()
            else:
                data_dict = data

            # Save incrementally to JSONL (one object per line)
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            
            print(f"[{i+1}/{len(all_links)}] Successfully scraped and saved: {url}")
            
        except Exception as e:
            print(f"[{i+1}/{len(all_links)}] Failed to scrape {url}: {e}")
            
    print(f"All processing complete. Data saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
