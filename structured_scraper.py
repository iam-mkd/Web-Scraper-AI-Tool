import re
import asyncio
import urllib.parse
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from scraper import fetch_html
from models import Product


async def search_amazon_products(
    query: str,
    timeout: float = 25.0,
    max_attempts: int = 3,
) -> List[Dict[str, Any]]:
    """
    Searches Amazon India (amazon.in) for the given query and extracts
    structured product information (name, price, capacity_gb, speed_mhz, cl_latency, rating, review_count, form_factor, kit_size, url).
    Bypasses anti-bot perimeter defenses using browser TLS impersonation with automatic retry backoff.
    Filters out non-RAM accessories (thermal pads, cables, etc.).
    """
    encoded_query = urllib.parse.quote_plus(query)
    search_url = f"https://www.amazon.in/s?k={encoded_query}"

    for attempt in range(1, max_attempts + 1):
        try:
            html = await fetch_html(search_url, timeout=timeout)
            soup = BeautifulSoup(html, "html.parser")

            # Match search result cards
            items = soup.select('div[data-component-type="s-search-result"]')
            if not items:
                # Fallback to general product items with ASINs
                items = [it for it in soup.select(".s-result-item[data-asin]") if it.get("data-asin")]

            products: List[Dict[str, Any]] = []

            for item in items:
                # Title
                h2 = item.find("h2")
                if not h2:
                    continue
                name = h2.get_text(separator=" ", strip=True)
                if not name:
                    continue

                name_lower = name.lower()

                # Price extraction (clean INR float)
                price_tag = (
                    item.select_one("span.a-price span.a-offscreen")
                    or item.select_one("span.a-price-whole")
                )
                price: Optional[float] = None
                if price_tag:
                    price_raw = price_tag.get_text(strip=True).split(".")[0]
                    price_clean = re.sub(r"[^\d]", "", price_raw)
                    if price_clean:
                        try:
                            price = float(price_clean)
                        except ValueError:
                            price = None

                # Rating extraction (float, e.g. 4.6)
                rating_tag = item.select_one("span.a-icon-alt")
                rating: Optional[float] = None
                if rating_tag:
                    m = re.search(r"(\d+(?:\.\d+)?)", rating_tag.get_text())
                    if m:
                        try:
                            rating = float(m.group(1))
                        except ValueError:
                            rating = None

                # Review count extraction (integer, e.g. 1243)
                reviews_tag = (
                    item.select_one("span.a-size-base.s-underline-text")
                    or item.select_one('a[href*="#customerReviews"] span')
                )
                review_count: Optional[int] = None
                if reviews_tag:
                    rev_clean = re.sub(r"[^\d]", "", reviews_tag.get_text())
                    if rev_clean:
                        try:
                            review_count = int(rev_clean)
                        except ValueError:
                            review_count = None

                # Clean canonical product URL
                a_tag = (
                    item.select_one("h2 a")
                    or item.select_one("a.a-link-normal.s-no-outline")
                    or item.select_one("a[href*='/dp/']")
                )
                product_url: Optional[str] = None
                if a_tag and a_tag.get("href"):
                    raw_href = a_tag["href"]
                    if raw_href.startswith("/"):
                        raw_href = f"https://www.amazon.in{raw_href}"
                    dp_match = re.search(r"(https://www\.amazon\.in/[^?]+/dp/[A-Z0-9]+)", raw_href)
                    if dp_match:
                        product_url = dp_match.group(1)
                    else:
                        product_url = raw_href.split("?")[0]

                # Extract RAM Kit Size and Capacity (GB)
                kit_match = re.search(r"\b(\d+)\s*[xX*]\s*(\d+)\s*(?:GB|gb)\b", name)
                cap_match = re.search(r"\b(\d+)\s*(?:GB|gb)\b", name)

                capacity_gb: Optional[int] = None
                kit_size: Optional[str] = None

                if kit_match:
                    modules = int(kit_match.group(1))
                    mod_size = int(kit_match.group(2))
                    kit_size = f"{modules}x{mod_size}GB"
                    capacity_gb = modules * mod_size
                elif cap_match:
                    capacity_gb = int(cap_match.group(1))
                    kit_size = f"1x{capacity_gb}GB"

                # Sanity check capacity: RAM modules typically 4GB to 128GB
                if capacity_gb is not None and (capacity_gb < 4 or capacity_gb > 256):
                    capacity_gb = None

                # Extract Form Factor (UDIMM vs SODIMM)
                form_factor: Optional[str] = None
                if any(k in name_lower for k in ["sodimm", "so-dimm", "laptop", "notebook"]):
                    form_factor = "SODIMM"
                elif any(k in name_lower for k in ["udimm", "u-dimm", "desktop", "dimm", "288-pin"]):
                    form_factor = "UDIMM"
                else:
                    # Default to UDIMM for desktop gaming memory kits
                    form_factor = "UDIMM"

                # Extract RAM speed (MHz or MT/s) if present
                speed_match = re.search(r"\b(\d{4,5})\s*(?:MHz|MT/s|mhz|mt/s)\b", name)
                speed_mhz: Optional[int] = int(speed_match.group(1)) if speed_match else None

                # Extract CAS Latency (e.g. CL30, CL36, CL40, C30)
                cl_match = re.search(r"\b(?:CL|C)\s*(\d{2})\b", name, re.IGNORECASE)
                cl_latency: Optional[int] = int(cl_match.group(1)) if cl_match else None

                # Discard non-RAM items (must have price, URL, and valid RAM capacity)
                if price is not None and product_url is not None and capacity_gb is not None:
                    prod_model = Product(
                        name=name,
                        price=price,
                        capacity_gb=capacity_gb,
                        speed_mhz=speed_mhz,
                        cl_latency=cl_latency,
                        rating=rating,
                        review_count=review_count,
                        form_factor=form_factor,
                        kit_size=kit_size,
                        url=product_url,
                    )
                    data = prod_model.model_dump()
                    data["reviews"] = review_count  # Backwards compatibility alias
                    products.append(data)

            if products:
                return products

            if attempt < max_attempts:
                await asyncio.sleep(1.2 * attempt)
        except Exception:
            if attempt < max_attempts:
                await asyncio.sleep(1.2 * attempt)
            else:
                raise

    return []


async def get_amazon_product_details(
    url: str,
    timeout: float = 25.0,
    max_attempts: int = 3,
) -> Dict[str, Any]:
    """
    Scrapes an Amazon product detail page and extracts granular hardware specifications:
    - Title, price, rating, reviews
    - Voltage, CAS latency, speed, form factor, kit size
    - Technical specifications table (Brand, Model, Dimensions, Warranty, etc.)
    - Key feature highlights
    Returns a clean, structured dictionary suitable for LLM reasoning and comparison.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            html = await fetch_html(url, timeout=timeout)
            soup = BeautifulSoup(html, "html.parser")

            # Title
            title_tag = soup.select_one("#productTitle") or soup.find("h1")
            title = title_tag.get_text(strip=True) if title_tag else "Unknown Product"
            title_lower = title.lower()

            # Price
            price_tag = (
                soup.select_one("span.apexPriceToPay span.a-offscreen")
                or soup.select_one("span.a-price span.a-offscreen")
                or soup.select_one("#priceblock_ourprice")
                or soup.select_one("#priceblock_dealprice")
            )
            price: Optional[float] = None
            if price_tag:
                raw_prc = price_tag.get_text(strip=True).split(".")[0]
                clean_prc = re.sub(r"[^\d]", "", raw_prc)
                if clean_prc:
                    try:
                        price = float(clean_prc)
                    except ValueError:
                        price = None

            # Rating
            rating_tag = soup.select_one("#acrPopover") or soup.select_one("span.a-icon-alt")
            rating: Optional[float] = None
            if rating_tag:
                m = re.search(r"(\d+(?:\.\d+)?)", rating_tag.get_text())
                if m:
                    try:
                        rating = float(m.group(1))
                    except ValueError:
                        rating = None

            # Review count
            review_tag = soup.select_one("#acrCustomerReviewText") or soup.select_one("#acrCustomerReviewLink")
            review_count: Optional[int] = None
            if review_tag:
                rev_clean = re.sub(r"[^\d]", "", review_tag.get_text())
                if rev_clean:
                    try:
                        review_count = int(rev_clean)
                    except ValueError:
                        review_count = None

            # Tech specs table
            raw_specs: Dict[str, str] = {}
            for row in soup.select("#productDetails_techSpec_section_1 tr, #tech_spec_section_1 tr, .prodDetTable tr"):
                th = row.find("th")
                td = row.find("td")
                if th and td:
                    k = th.get_text(strip=True).replace("\u200e", "").replace("\u200f", "")
                    v = td.get_text(strip=True).replace("\u200e", "").replace("\u200f", "")
                    if k and v:
                        raw_specs[k] = v

            # Feature bullets (clean, limit to top 4)
            bullets: List[str] = []
            for li in soup.select("#feature-bullets ul li span.a-list-item"):
                b_text = li.get_text(strip=True)
                if b_text and len(b_text) > 15 and not any(skip in b_text.lower() for skip in ["delivery", "return policy", "customer service"]):
                    bullets.append(b_text)
            highlights = bullets[:4]

            # Brand & Model
            brand = raw_specs.get("Brand") or raw_specs.get("Manufacturer")
            if not brand:
                for b_cand in ["Corsair", "Crucial", "Kingston", "G.Skill", "Teamgroup", "Patriot", "ADATA", "XPG", "Samsung"]:
                    if b_cand.lower() in title_lower:
                        brand = b_cand
                        break

            model_number = raw_specs.get("Item model number") or raw_specs.get("Model")

            # Voltage extraction (e.g. 1.25V, 1.35V, 1.4V, 1.1V)
            voltage: Optional[str] = raw_specs.get("Voltage")
            if not voltage:
                v_match = re.search(r"\b(1\.\d{1,2}\s*[vV])\b", title)
                if not v_match:
                    for b in highlights:
                        v_match = re.search(r"\b(1\.\d{1,2}\s*[vV])\b", b)
                        if v_match:
                            break
                if v_match:
                    voltage = v_match.group(1).upper()

            # Warranty extraction
            warranty: Optional[str] = raw_specs.get("Warranty") or raw_specs.get("Manufacturer Warranty")
            if not warranty:
                for b in highlights:
                    if "warranty" in b.lower():
                        warranty = b[:80]
                        break

            # Speed extraction
            speed_str = raw_specs.get("Memory Speed") or raw_specs.get("RAM Memory Technology") or ""
            speed_match = re.search(r"(\d{4,5})", speed_str) or re.search(r"\b(\d{4,5})\s*(?:MHz|MT/s|mhz|mt/s)\b", title)
            speed_mhz: Optional[int] = int(speed_match.group(1)) if speed_match else None

            # Capacity and Kit Size
            kit_match = re.search(r"\b(\d+)\s*[xX*]\s*(\d+)\s*(?:GB|gb)\b", title)
            cap_match = re.search(r"\b(\d+)\s*(?:GB|gb)\b", title)
            capacity_gb: Optional[int] = None
            kit_size: Optional[str] = None
            if kit_match:
                modules = int(kit_match.group(1))
                mod_size = int(kit_match.group(2))
                kit_size = f"{modules}x{mod_size}GB"
                capacity_gb = modules * mod_size
            elif cap_match:
                capacity_gb = int(cap_match.group(1))
                kit_size = f"1x{capacity_gb}GB"

            # Form Factor
            form_factor = raw_specs.get("Form Factor")
            if not form_factor:
                if any(k in title_lower for k in ["sodimm", "so-dimm", "laptop", "notebook"]):
                    form_factor = "SODIMM"
                else:
                    form_factor = "UDIMM"

            # CAS Latency
            cl_match = re.search(r"\b(?:CL|C)\s*(\d{2})\b", title, re.IGNORECASE)
            cl_latency: Optional[int] = int(cl_match.group(1)) if cl_match else None

            # Filter specs to most relevant compact subset for LLM context
            relevant_specs = {
                k: v for k, v in raw_specs.items()
                if any(kw in k.lower() for kw in ["brand", "model", "speed", "voltage", "dimension", "form factor", "pin", "weight", "warranty"])
            }
            if not relevant_specs and raw_specs:
                relevant_specs = dict(list(raw_specs.items())[:8])

            return {
                "name": title,
                "price": price,
                "capacity_gb": capacity_gb,
                "speed_mhz": speed_mhz,
                "cl_latency": cl_latency,
                "voltage": voltage,
                "warranty": warranty,
                "form_factor": form_factor,
                "kit_size": kit_size,
                "brand": brand,
                "model_number": model_number,
                "rating": rating,
                "review_count": review_count,
                "tech_specs": relevant_specs,
                "highlights": highlights,
                "url": url,
            }

        except Exception:
            if attempt < max_attempts:
                await asyncio.sleep(1.2 * attempt)
            else:
                return {
                    "name": "Product Details Extraction Error",
                    "url": url,
                    "error": f"Failed to retrieve product details after {max_attempts} attempts",
                    "tech_specs": {},
                    "highlights": [],
                }

    return {"name": "Product Details Unavailable", "url": url}


