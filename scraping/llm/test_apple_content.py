import asyncio
from playwright.async_api import async_playwright

APPLE_URL = "http://www.gardenology.org/wiki/Apple"

async def test_apple_content():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f"Loading {APPLE_URL}...")
        await page.goto(APPLE_URL)
        
        # Exact logic from the filter script
        result = await page.evaluate('''() => {
            const headline = document.getElementById("Pests_and_diseases");
            if (!headline) return { found: false, reason: "No ID found" };
            
            const parentHeading = headline.parentElement;
            let nextElem = parentHeading.nextElementSibling;
            
            const results = [];
            while (nextElem && !['H1', 'H2', 'H3', 'H4', 'H5', 'H6'].includes(nextElem.tagName)) {
                const text = nextElem.innerText.trim();
                const isPlaceholder = text.includes("Edit this section!") || 
                                    text.includes("Do you have pest and disease info") ||
                                    text.includes("Edit this page");
                
                results.push({ tag: nextElem.tagName, text: text.substring(0, 50), isPlaceholder });
                
                if (!isPlaceholder && text.length > 30) { 
                    return { found: true, results };
                }
                nextElem = nextElem.nextElementSibling;
            }
            return { found: false, reason: "No content after heading", results };
        }''')
        
        print(f"Result for Apple: {result}")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_apple_content())
