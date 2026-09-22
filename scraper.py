import re
from typing import Dict, Any
from bs4 import BeautifulSoup

try:
    from curl_cffi.requests import AsyncSession as CurlAsyncSession
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False

import httpx


async def fetch_html(url: str, timeout: float = 25.0) -> str:
    """
    Fetches HTML content with browser TLS/JA3 fingerprint impersonation
    using curl_cffi to bypass anti-bot shields (Amazon, Cloudflare, etc.).
    Falls back to httpx if curl_cffi is unavailable.
    """
    if CURL_CFFI_AVAILABLE:
        try:
            async with CurlAsyncSession(impersonate="chrome120") as session:
                response = await session.get(url, timeout=timeout)
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 999 or "linkedin.com" in url.lower() and response.status_code != 200:
                    raise PermissionError(
                        "LinkedIn blocked access with status code 999 (Request Denied). "
                        "LinkedIn personal profiles require an authenticated session (e.g., 'li_at' cookie) "
                        "or official LinkedIn API access because they are protected behind a login wall."
                    )
                elif response.status_code in (403, 503):
                    # Try another impersonation target if blocked
                    async with CurlAsyncSession(impersonate="safari15_5") as safari_session:
                        resp_safari = await safari_session.get(url, timeout=timeout)
                        if resp_safari.status_code == 200:
                            return resp_safari.text
                response.raise_for_status()
                return response.text
        except PermissionError:
            raise
        except Exception:
            # Fallback to httpx if curl_cffi encounters an internal network issue
            pass

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/128.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-User": "?1",
        "Sec-Fetch-Dest": "document",
    }

    async with httpx.AsyncClient(headers=headers, follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url)
        if response.status_code == 999 or ("linkedin.com" in url.lower() and response.status_code != 200):
            raise PermissionError(
                "LinkedIn blocked access with status code 999 (Request Denied). "
                "LinkedIn personal profiles require an authenticated session (e.g., 'li_at' cookie) "
                "or official LinkedIn API access because they are protected behind a login wall."
            )
        response.raise_for_status()
        return response.text


async def scrape_webpage(url: str, timeout: float = 25.0) -> Dict[str, Any]:
    """
    Scrapes a webpage, extracts title, strips boilerplate, and cleans the text.
    """
    html_content = await fetch_html(str(url), timeout=timeout)
    soup = BeautifulSoup(html_content, "html.parser")

    # Extract page title
    title = soup.title.string.strip() if (soup.title and soup.title.string) else "Untitled Webpage"

    # Remove non-content tags and common noise containers (sidebars, footers, ad facets)
    noise_selectors = [
        "script", "style", "nav", "footer", "header", "aside", "noscript", "svg", "form",
        "#navFooter", "#s-refinements", "#rhf", "#nav-subnav", "#nav-upnav",
        ".s-desktop-toolbar", ".s-breadcrumb", ".nav-footer", ".nav-subnav",
        ".sidebar", ".widget-area", ".breadcrumb", ".breadcrumbs", ".pagination",
        ".cookie-banner", ".cookie-notice", ".social-share", ".advertisement", ".ad-container",
        "[role='navigation']", "[role='banner']", "[role='contentinfo']",
    ]
    for sel in noise_selectors:
        for tag in soup.select(sel):
            tag.decompose()

    # Prioritize main content if available
    main_content = soup.find("article") or soup.find("main") or soup.body or soup

    # Extract clean text
    raw_text = main_content.get_text(separator="\n", strip=True)

    # Normalize whitespace: replace multiple blank lines with double newlines
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", raw_text).strip()

    if not cleaned_text:
        raise ValueError("Could not extract any meaningful text from the provided webpage.")

    return {
        "url": str(url),
        "title": title,
        "content": cleaned_text,
        "char_count": len(cleaned_text),
    }
