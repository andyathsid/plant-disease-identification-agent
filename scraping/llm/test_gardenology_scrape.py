import json
import os
import asyncio
import logging
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
    
    return smart_scraper.run()

async def main():
    # Test with only the first URL
    url = "http://www.gardenology.org/wiki/Celery"
    
    try:
        data = await asyncio.to_thread(scrape_gardenology_details, url)
        
        if isinstance(data, BaseModel):
            data_dict = data.model_dump()
        else:
            data_dict = data

        print("\nScraped Data Result:")
        print(json.dumps(data_dict, indent=2))
        
    except Exception as e:
        print(f"Failed to scrape {url}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
