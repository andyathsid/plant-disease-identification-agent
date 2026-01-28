import asyncio
from urllib.parse import urljoin
from playwright.async_api import async_playwright

BASE_URL = "http://www.gardenology.org"
FRUITS_LIST = urljoin(BASE_URL, "/wiki/List_of_fruits")

async def test_apple_discovery():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        print(f"Loading {FRUITS_LIST}...")
        await page.goto(FRUITS_LIST)
        
        links = await page.eval_on_selector_all(
            '#mw-content-text a:not(.new)',
            '''nodes => nodes
                .map(n => ({ href: n.getAttribute("href"), title: n.getAttribute("title"), text: n.innerText }))
                .filter(n => {
                    if (!n.href || !n.href.startsWith("/wiki/")) return false;
                    const path = n.href.split("/wiki/")[1];
                    if (!path) return false;
                    if (path.includes(":") || path.includes("Main_Page") || path.startsWith("List_of_")) return false;
                    return true;
                })'''
        )
        
        print(f"Total links found: {len(links)}")
        apple_link = next((l for l in links if l['href'] == '/wiki/Apple'), None)
        if apple_link:
            print(f"Found Apple link: {apple_link}")
        else:
            print("Apple link NOT found in the list.")
            # Print first 10 links for debugging
            print("First 10 links found:")
            for l in links[:10]:
                print(l)
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_apple_discovery())
