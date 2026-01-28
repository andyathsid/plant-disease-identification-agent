import asyncio
import json
import os
from playwright.async_api import async_playwright

INPUT_FILE = "llm/data/gardenology_relevant_links.jsonl"
OUTPUT_FILE = "llm/data/gardenology_verified_links.jsonl"

async def is_truly_relevant(browser, url):
    """Stricter check to filter out 'Edit this section!' placeholders with retries."""
    for attempt in range(3):
        page = await browser.new_page()
        try:
            # Add a bit of delay between retries
            if attempt > 0:
                await asyncio.sleep(2 * attempt)
            
            await page.goto(url, timeout=60000)
            
            # Stricter JS check
            result = await page.evaluate('''() => {
                const headline = document.getElementById("Pests_and_diseases");
                if (!headline) return { hasContent: false, reason: "No Pests_and_diseases ID" };
                
                // Get the parent element (usually H2 or H3)
                const parentHeading = headline.parentElement;
                let nextElem = parentHeading.nextElementSibling;
                
                let combinedText = "";
                // Traverse until next heading
                while (nextElem && !['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(nextElem.tagName)) {
                    // Check if it's a "printfooter" or similar meta-div
                    if (nextElem.tagName === 'DIV' && nextElem.classList.contains('printfooter')) break;
                    
                    combinedText += nextElem.innerText + " ";
                    nextElem = nextElem.nextElementSibling;
                }
                
                const text = combinedText.trim();
                const isPlaceholder = text.includes("Edit this section!") || 
                                    text.includes("Do you have pest and disease info") ||
                                    text.includes("Edit this page") ||
                                    text.length < 20;
                
                return { hasContent: !isPlaceholder, textLength: text.length };
            }''')
            
            return result['hasContent']
        except Exception as e:
            if attempt == 2:
                print(f"Failed to check {url} after 3 attempts: {e}")
        finally:
            await page.close()
    return False

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Input file {INPUT_FILE} not found.")
        return

    # Read existing links
    links_to_check = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            data = json.loads(line)
            links_to_check.append(data["url"])

    print(f"Filtering {len(links_to_check)} links...")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        
        verified_count = 0
        processed_count = 0
        total = len(links_to_check)
        
        semaphore = asyncio.Semaphore(10)
        
        async def worker(url):
            nonlocal verified_count, processed_count
            async with semaphore:
                if await is_truly_relevant(browser, url):
                    with open(OUTPUT_FILE, "a") as out_f:
                        out_f.write(json.dumps({"url": url}) + "\n")
                    verified_count += 1
                
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Progress: {processed_count}/{total} (Verified: {verified_count})")

        tasks = [worker(url) for url in links_to_check]
        await asyncio.gather(*tasks)
        
        await browser.close()

    print(f"Done! {verified_count} links verified and saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())
