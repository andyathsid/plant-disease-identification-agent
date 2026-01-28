import asyncio
import json
import os
from urllib.parse import urljoin
from playwright.async_api import async_playwright

BASE_URL = "http://www.gardenology.org"
MAIN_PAGE = urljoin(BASE_URL, "/wiki/Main_Page")

async def get_category_links(page):
    """Extracts links to 'List of ...' pages from the Main Page."""
    print(f"Fetching category links from {MAIN_PAGE}...")
    await page.goto(MAIN_PAGE)
    
    # Based on base.html, the category links are in the 'Plant Categories' section
    # They are typically links starting with /wiki/List_of_
    links = await page.eval_on_selector_all(
        '#mw-content-text table a',
        'nodes => nodes.map(n => n.getAttribute("href")).filter(h => h && h.includes("/wiki/List_of_"))'
    )
    
    unique_links = sorted(list(set(urljoin(BASE_URL, link) for link in links)))
    print(f"Found {len(unique_links)} category list links.")
    return unique_links

async def get_plant_links(page, category_url):
    """Extracts individual plant links from a category list page."""
    print(f"Fetching plant links from {category_url}...")
    await page.goto(category_url)
    
    # Extract all wiki links that are NOT:
    # - Redlinks (class 'new')
    # - Special pages, Files, Talk, Category, etc.
    # - List pages themselves
    links = await page.eval_on_selector_all(
        '#mw-content-text a:not(.new)',
        '''nodes => nodes
            .map(n => ({ href: n.getAttribute("href"), title: n.getAttribute("title") }))
            .filter(n => {
                if (!n.href || !n.href.startsWith("/wiki/")) return false;
                const path = n.href.split("/wiki/")[1];
                if (!path) return false;
                // Filter out non-plant namespaces and meta pages
                if (path.includes(":") || path.includes("Main_Page") || path.startsWith("List_of_")) return false;
                return true;
            })
            .map(n => n.href)'''
    )
    
    unique_links = sorted(list(set(urljoin(BASE_URL, link) for link in links)))
    print(f"Found {len(unique_links)} potential plant links in {category_url}.")
    return unique_links

async def has_pests_and_diseases(browser, plant_url):
    """Checks if a plant page has a 'Pests and diseases' section with actual content."""
    page = await browser.new_page()
    try:
        await page.goto(plant_url, timeout=60000)
        
        # Stricter check for content
        has_content = await page.evaluate('''() => {
            const headline = document.getElementById("Pests_and_diseases");
            if (!headline) return false;
            
            const parentHeading = headline.parentElement;
            let nextElem = parentHeading.nextElementSibling;
            
            while (nextElem && !['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(nextElem.tagName)) {
                const text = nextElem.innerText.trim();
                
                // Filter out the "Edit this section!" placeholders
                const isPlaceholder = text.includes("Edit this section!") || 
                                    text.includes("Do you have pest and disease info") ||
                                    text.includes("Edit this page");
                
                if (!isPlaceholder && text.length > 30) { 
                    return true;
                }
                nextElem = nextElem.nextElementSibling;
            }
            return false;
        }''')
        
        return has_content
    except Exception:
        pass
    finally:
        await page.close()
    return False

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # Initial page to get links
        temp_page = await browser.new_page()
        category_links = await get_category_links(temp_page)
        
        all_plant_urls = set()
        for cat_link in category_links:
            plant_links = await get_plant_links(temp_page, cat_link)
            all_plant_urls.update(plant_links)
        
        await temp_page.close()
        
        print(f"Total unique plant URLs found: {len(all_plant_urls)}")
        
        output_file = "llm/data/gardenology_relevant_links.jsonl"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        relevant_plants = []
        semaphore = asyncio.Semaphore(10) # 10 concurrent pages
        processed_count = 0
        total_count = len(all_plant_urls)

        async def worker(url):
            nonlocal processed_count
            async with semaphore:
                if await has_pests_and_diseases(browser, url):
                    with open(output_file, "a") as f:
                        f.write(json.dumps({"url": url}) + "\n")
                    relevant_plants.append(url)
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Progress: {processed_count}/{total_count} (Found: {len(relevant_plants)})")

        tasks = [worker(url) for url in all_plant_urls]
        await asyncio.gather(*tasks)

        await browser.close()
        
    print(f"Done! Found {len(relevant_plants)} relevant plants. Links saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())
