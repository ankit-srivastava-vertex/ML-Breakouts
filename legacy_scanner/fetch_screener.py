"""
Fetch stock names from a screener.in screen URL and write to Excel.

Usage:
  python fetch_screener.py
  python fetch_screener.py "https://www.screener.in/screens/2877406/52w-15/"
"""

import os
import sys
import re
import urllib.request
import urllib.parse
import http.cookiejar

from dotenv import load_dotenv
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGIN_URL = "https://www.screener.in/login/"
DEFAULT_SCREEN_URL = "https://www.screener.in/screens/2877406/52w-15/"

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _build_opener():
    cj = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))


def _get(opener, url, referer=None):
    headers = {"User-Agent": UA, "Accept": "text/html"}
    if referer:
        headers["Referer"] = referer
    req = urllib.request.Request(url, headers=headers)
    return opener.open(req, timeout=30).read().decode("utf-8", errors="ignore")


def _post(opener, url, data, referer=None):
    headers = {"User-Agent": UA, "Content-Type": "application/x-www-form-urlencoded"}
    if referer:
        headers["Referer"] = referer
    encoded = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=encoded, headers=headers)
    return opener.open(req, timeout=30).read().decode("utf-8", errors="ignore")


def login(opener, email, password):
    """Log in to screener.in. Returns True on success."""
    html = _get(opener, LOGIN_URL)
    m = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', html)
    if not m:
        print("ERROR: Could not find CSRF token on login page.", file=sys.stderr)
        return False
    csrf = m.group(1)

    body = _post(opener, LOGIN_URL, {
        "csrfmiddlewaretoken": csrf,
        "username": email,
        "password": password,
    }, referer=LOGIN_URL)

    if "Please enter a correct" in body:
        print("ERROR: Login failed — invalid credentials.", file=sys.stderr)
        return False
    return True


def fetch_screen(opener, url):
    """Fetch all pages of a screener.in screen. Returns list of stock names."""
    names = []
    page = 1
    while True:
        page_url = f"{url.rstrip('/')}/?page={page}" if page > 1 else url
        html = _get(opener, page_url, referer="https://www.screener.in/")
        found = re.findall(
            r'href="/company/[^/]+/[^"]*"[^>]*>\s*([^<]+?)\s*</a>', html)
        if not found:
            break
        names.extend(found)
        page += 1
        if f"page={page}" not in html and "Next" not in html:
            break
    return names


def main():
    load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

    email = os.environ.get("SCREENER_USER", "")
    password = os.environ.get("SCREENER_PASS", "")
    if not email or not password:
        print("ERROR: Set SCREENER_USER and SCREENER_PASS in .env",
              file=sys.stderr)
        sys.exit(1)

    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SCREEN_URL

    opener = _build_opener()

    print("Logging in to screener.in ...", file=sys.stderr)
    if not login(opener, email, password):
        sys.exit(1)
    print("Login OK.", file=sys.stderr)

    print(f"Fetching screen: {url}", file=sys.stderr)
    names = fetch_screen(opener, url)

    if not names:
        print("No stocks found (check URL or screen visibility).",
              file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(names)} stocks.", file=sys.stderr)

    # Derive output filename from screen URL slug
    slug = url.rstrip("/").split("/")[-1] or "screen"
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "Output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "screener_data.xlsx")

    df = pd.DataFrame({"Name": names})
    df.to_excel(out_path, index=False, engine="openpyxl")
    print(f"Excel written: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
