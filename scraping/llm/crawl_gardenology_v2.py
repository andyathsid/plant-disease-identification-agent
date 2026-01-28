import asyncio
import json
import os
from urllib.parse import urljoin
from playwright.async_api import async_playwright

BASE_URL = "http://www.gardenology.org"
MAIN_PAGE = urljoin(BASE_URL, "/wiki/Main_Page")
OUTPUT_FILE = "llm/data/gardenology_verified_links_v2.jsonl"

async def get_category_links(page):
    print(f"Fetching category links from {MAIN_PAGE}...")
    for attempt in range(3):
        try:
            await page.goto(MAIN_PAGE, timeout=60000)
            links = await page.eval_on_selector_all(
                '#mw-content-text table a',
                'nodes => nodes.map(n => n.getAttribute("href")).filter(h => h && h.includes("/wiki/List_of_"))'
            )
            return sorted(list(set(urljoin(BASE_URL, link) for link in links)))
        except Exception as e:
            print(f"Attempt {attempt+1} failed for Main Page: {e}")
            await asyncio.sleep(2)
    return []

async def get_plant_links(browser, category_url):
    page = await browser.new_page()
    try:
        print(f"Fetching plant links from {category_url}...")
        for attempt in range(3):
            try:
                await page.goto(category_url, timeout=60000)
                links = await page.eval_on_selector_all(
                    '#mw-content-text a:not(.new)',
                    '''nodes => nodes
                        .map(n => ({ href: n.getAttribute("href") }))
                        .filter(n => {
                            if (!n.href || !n.href.startsWith("/wiki/")) return false;
                            const path = n.href.split("/wiki/")[1];
                            if (!path || path.includes(":") || path.includes("Main_Page") || path.startsWith("List_of_")) return false;
                            return true;
                        })
                        .map(n => n.href)'''
                )
                return sorted(list(set(urljoin(BASE_URL, link) for link in links)))
            except Exception as e:
                print(f"Attempt {attempt+1} failed for {category_url}: {e}")
                await asyncio.sleep(2)
    finally:
        await page.close()
    return []

async def is_truly_relevant(browser, url):
    page = await browser.new_page()
    try:
        for attempt in range(3):
            try:
                await page.goto(url, timeout=60000)
                result = await page.evaluate('''() => {
                    const headline = document.getElementById("Pests_and_diseases");
                    if (!headline) return false;
                    
                    const parentHeading = headline.parentElement;
                    let nextElem = parentHeading.nextElementSibling;
                    let combinedText = "";
                    
                    while (nextElem && !['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(nextElem.tagName)) {
                        if (nextElem.tagName === 'DIV' && nextElem.classList.contains('printfooter')) break;
                        combinedText += nextElem.innerText + " ";
                        nextElem = nextElem.nextElementSibling;
                    }
                    
                    const text = combinedText.trim();
                    const isPlaceholder = text.includes("Edit this section!") || 
                                        text.includes("Do you have pest and disease info") ||
                                        text.includes("Edit this page") ||
                                        text.length < 30;
                    
                    return !isPlaceholder;
                }''')
                return result
            except Exception:
                await asyncio.sleep(1)
    finally:
        await page.close()
    return False

async def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        # 1. Get Categories
        temp_page = await browser.new_page()
        category_links = await get_category_links(temp_page)
        await temp_page.close()
        
        if not category_links:
            print("No categories found.")
            await browser.close()
            return

        # 2. Get all plant URLs
        all_plant_urls = set()
        for cat_url in category_links:
            plant_links = await get_plant_links(browser, cat_url)
            all_plant_urls.update(plant_links)
        
        print(f"Total unique plant URLs found: {len(all_plant_urls)}")
        
        # 3. Filter and save
        semaphore = asyncio.Semaphore(5) # Lowered concurrency for stability
        processed_count = 0
        verified_count = 0
        total_count = len(all_plant_urls)

        async def worker(url):
            nonlocal processed_count, verified_count
            async with semaphore:
                if await is_truly_relevant(browser, url):
                    with open(OUTPUT_FILE, "a") as f:
                        f.write(json.dumps({"url": url}) + "\n")
                    verified_count += 1
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Progress: {processed_count}/{total_count} (Verified: {verified_count})")

        tasks = [worker(url) for url in all_plant_urls]
        await asyncio.gather(*tasks)
        
        await browser.close()
    
    print(f"Done! Verified {verified_count} plants in {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
