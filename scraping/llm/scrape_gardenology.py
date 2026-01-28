import json
import os
import asyncio
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from scrapegraphai.graphs import SmartScraperGraph

# Load environment variables
load_dotenv()

# Silence ScrapeGraphAI
logging.getLogger("scrapegraphai").setLevel(logging.WARNING)

# 1. Define the schema based on gardenology_structure.json
class GardenologyPlantInfo(BaseModel):
    plant: str
    description: str
    cultivation: str
    propagation: str
    pests_and_diseases: str

# 2. Configure the graph
graph_config = {
    "llm": {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "model": "google_genai/gemini-2.5-flash", 
        "temperature": 0.0,
        "max_tokens": 8192,
    },
    "verbose": False,
    "headless": True,
}

def scrape_gardenology_details(url: str):
    """Scrapes a single Gardenology wiki page using the defined schema."""
    print(f"Scraping details for: {url}")
    
    prompt = (
        "Extract the plant name, a general description, cultivation details, "
        "propagation methods, and information about pests and diseases. "
        "If a section is missing, use an empty string."
    )
    
    smart_scraper = SmartScraperGraph(
        prompt=prompt,
        source=url,
        config=graph_config,
        schema=GardenologyPlantInfo
    )
    
    # Use redirect_stdout/stderr to catch any internal prints from the library
    with open(os.devnull, 'w') as fnull:
        with redirect_stdout(fnull), redirect_stderr(fnull):
            return smart_scraper.run()

async def main():
    input_path = os.path.join("llm", "data", "gardenology_verified_links_v2.jsonl")
    output_path = os.path.join("llm", "data", "gardenology_content.jsonl")
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load links from the verified links file
    links = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                links.append(json.loads(line)["url"])

    print(f"Starting extraction for {len(links)} plants...")
    
    for i, url in enumerate(links):
        try:
            # Run the sync scraper in a thread
            data = await asyncio.to_thread(scrape_gardenology_details, url)
            
            if isinstance(data, BaseModel):
                data_dict = data.model_dump()
            else:
                data_dict = data

            # Save incrementally
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data_dict, ensure_ascii=False) + "\n")
            
            print(f"[{i+1}/{len(links)}] Successfully scraped: {url}")
            
        except Exception as e:
            print(f"[{i+1}/{len(links)}] Failed to scrape {url}: {e}")
            
    print(f"All processing complete. Data saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
