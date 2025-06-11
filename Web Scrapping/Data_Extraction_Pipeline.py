# Ingram Micro Product Categorization Project - Data Extraction Pipeline
# Prerequisites (run in terminal before first use):
# pip install playwright beautifulsoup4 pandas openpyxl requests
# playwright install

import asyncio
import pandas as pd
import os
import time
from bs4 import BeautifulSoup
import requests

# === CDW EXTRACTION ===
async def run_cdw_extraction(input_file="manufacturer_part_id.xlsx", autosave_freq=10, output_file="updated_cdw_results_urls_test.xlsx"):
    from playwright.async_api import async_playwright
    df_input = pd.read_excel(input_file)
    original_parts = df_input['manufacturer_part_nbr'].dropna().astype(str).tolist()

    results = []
    os.makedirs("html_logs", exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for idx, mfg in enumerate(original_parts):
            print(f"\nSearching CDW: {mfg}")
            search_url = f"https://www.cdw.com/search/?key={mfg}"
            try:
                await page.goto(search_url, timeout=60000, wait_until="domcontentloaded")
            except Exception as e:
                print(f"Failed to load {search_url}: {e}")
                continue

            current_url = page.url
            product_urls = []

            if "/product/" in current_url:
                product_urls = [current_url]
            else:
                try:
                    await page.wait_for_selector('.search-results', timeout=10000)
                    await page.wait_for_timeout(2000)
                    links = await page.query_selector_all('a[href*="/product/"]')
                    seen = set()
                    for link in links:
                        href = await link.get_attribute('href')
                        if href and "/product/" in href:
                            full_url = f"https://www.cdw.com{href.split('?')[0]}"
                            if full_url not in seen:
                                seen.add(full_url)
                                product_urls.append(full_url)
                except Exception as e:
                    print(f"Failed to fetch product links for {mfg}: {e}")

            matched_url = None
            for i, url in enumerate(product_urls):
                try:
                    await page.goto(url, timeout=30000)
                    html = await page.content()
                    with open(f"html_logs/{mfg}_{i}.html", "w", encoding="utf-8") as f:
                        f.write(html)
                    if mfg.lower() in html.lower():
                        matched_url = url
                        print(f"MPN matched: {url}")
                        break
                    else:
                        print(f"No match for {mfg} in: {url}")
                except Exception as e:
                    print(f"Failed to validate product URL: {url} — {e}")

            results.append({
                "manufacturer_part_nbr": mfg,
                "product_url": matched_url
            })

            if (idx + 1) % autosave_freq == 0:
                pd.DataFrame(results).to_excel("autosave_cdw.xlsx", index=False)

        await browser.close()

    df_result = pd.DataFrame(results)
    df_result.to_excel(output_file, index=False)
    print(f"CDW extraction completed: {output_file}")
    return df_result

# === AMAZON EXTRACTION ===
def extract_amazon_data(mpn_list, output_file="amazon_product_links.xlsx"):
    results = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for mpn in mpn_list:
        print(f"Searching Amazon: {mpn}")
        search_url = f"https://www.amazon.com/s?k={mpn}"
        try:
            res = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(res.content, "html.parser")
            link_tag = soup.select_one("a.a-link-normal.s-no-outline")
            matched_url = f"https://www.amazon.com{link_tag['href'].split('?')[0]}" if link_tag else None
        except Exception as e:
            print(f"Error fetching Amazon data for {mpn}: {e}")
            matched_url = None

        results.append({
            "MPN": mpn,
            "Amazon Product URL": matched_url
        })

    df = pd.DataFrame(results, columns=["MPN", "Amazon Product URL"])
    df.to_excel(output_file, index=False)
    print(f"Amazon extraction completed: {output_file}")
    return df

# === GEMINI FETCH ===
GOOGLE_API_KEY = "AIzaSyAmmr-NxiLwUv8mxc1HrTIkQJckB8sE-90"
API_MODEL = "models/gemini-1.5-pro"
API_URL = f"https://generativelanguage.googleapis.com/v1/{API_MODEL}:generateContent?key={GOOGLE_API_KEY}"

def build_prompt_from_url(url):
    return f"""
    Visit and analyze the product page from this CDW URL: {url}

    Then generate structured product details as follows:
    - Product Title (max 6-7 words)
    - Brand Name
    - Size or Dimensions
    - Short Product Description (1-2 lines)
    - Detailed Product Description (technical or usage-focused)
    - Central Description (core product functionality or differentiator)
    - Summary (1–2 sentence marketing-style summary highlighting the product's value)

    Format each response as a clean new line (no bullets, no commentary).
    Focus only on enterprise-grade CDW IT products (hardware, software, AV, servers, cables, OS, firewalls, security).
    """

def get_google_gemini_data(prompt, max_retries=3, delay=5):
    headers = {"Content-Type": "application/json"}
    data = {"contents": [{"parts": [{"text": prompt}]}]}

    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, headers=headers, json=data, timeout=20)
            response.raise_for_status()
            json_response = response.json()
            if "candidates" in json_response:
                return json_response["candidates"][0]["content"]["parts"][0]["text"]
            return "Error: Unexpected response format"
        except requests.exceptions.RequestException as e:
            print(f"Error on attempt {attempt+1}: {e}")
            time.sleep(delay)

    return "Error: Maximum retries reached"

def fetch_structured_gemini_data(pairs, output_file="gemini_product_details_with_part_id.xlsx"):
    data = []
    for part_id, url in pairs:
        print(f"Processing: {part_id} | {url}")
        prompt = build_prompt_from_url(url)
        response_text = get_google_gemini_data(prompt)

        if "Error" in response_text:
            print(f"Skipping {part_id} due to error.")
            continue

        lines = [line.strip("-• ").strip() for line in response_text.strip().split("\n") if line.strip()]
        while len(lines) < 7:
            lines.append("Missing")

        record = {
            "mfr_part_number": part_id,
            "product_title": lines[0],
            "brand": lines[1],
            "size": lines[2],
            "product_desc_1": lines[3],
            "technical_details": lines[4],
            "central_description": lines[5],
            "summary": lines[6],
            "product_url": url
        }
        data.append(record)
        time.sleep(2)

    df = pd.DataFrame(data)
    df.to_excel(output_file, index=False)
    print(f"Gemini product info saved to {output_file}")
    return df

# === MAIN RUNNER ===
def run_pipeline(input_excel="manufacturer_part_id.xlsx"):
    print("Starting Data Extraction Pipeline...")

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    cdw_df = loop.run_until_complete(run_cdw_extraction(input_file=input_excel))
    cdw_df["product_url"] = cdw_df["product_url"].fillna("")

    mpns_missing = cdw_df[cdw_df["product_url"] == ""]["manufacturer_part_nbr"].tolist()
    amazon_df = extract_amazon_data(mpns_missing)

    amazon_dict = dict(zip(amazon_df["MPN"], amazon_df["Amazon Product URL"]))
    cdw_df["Amazon Product URL"] = cdw_df["manufacturer_part_nbr"].map(amazon_dict)

    cdw_df.to_excel("combined_cdw_amazon.xlsx", index=False)

    valid_pairs = cdw_df.dropna(subset=["product_url"])
    pairs = list(zip(valid_pairs["manufacturer_part_nbr"].astype(str), valid_pairs["product_url"].astype(str)))
    fetch_structured_gemini_data(pairs)

    print("Pipeline completed successfully.")

if __name__ == "__main__":
    run_pipeline()
