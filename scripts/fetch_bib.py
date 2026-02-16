#!/usr/bin/env python3
import argparse
import sys
import difflib
import requests
from habanero import Crossref

# --- Configuration ---
# Add your email here to get "polite" pool access (faster/reliable) from Crossref
EMAIL = "p.templier@imperial.ac.uk" 
# Similarity threshold (0.0 to 1.0) to consider a result an "exact match"
MATCH_THRESHOLD = 0.85

def normalize_title(title):
    """Normalize title for comparison (lowercase, remove simple punctuation)."""
    return "".join(c for c in title.lower() if c.isalnum() or c.isspace())

def is_match(query, result_title):
    """Check if the found title matches the query using sequence matching."""
    norm_query = normalize_title(query)
    norm_result = normalize_title(result_title)
    
    # Exact substring match (often sufficient)
    if norm_query in norm_result or norm_result in norm_query:
        return True
        
    # Fuzzy match ratio
    ratio = difflib.SequenceMatcher(None, norm_query, norm_result).ratio()
    return ratio >= MATCH_THRESHOLD

def get_bibtex_crossref(query):
    """
    Searches Crossref. Returns BibTeX string if a good match is found, else None.
    """
    cr = Crossref(mailto=EMAIL)
    try:
        # Search for the work
        results = cr.works(query=query, limit=1)
        items = results['message']['items']
        
        if not items:
            return None
            
        top_hit = items[0]
        title_list = top_hit.get('title', [])
        found_title = title_list[0] if title_list else "No Title"
        
        # Verify match quality
        if is_match(query, found_title):
            print(f"✅ Found on Crossref: {found_title}", file=sys.stderr)
            doi = top_hit.get('DOI')
            return cr.content_negotiation(ids=doi, format="bibtex")
        else:
            print(f"⚠️  Crossref mismatch (Query: '{query}' vs Found: '{found_title}')", file=sys.stderr)
            return None

    except Exception as e:
        print(f"❌ Crossref Error: {e}", file=sys.stderr)
        return None

def get_bibtex_dblp(query):
    """
    Searches DBLP. Returns BibTeX string if found, else None.
    """
    try:
        # DBLP API Search
        url = "https://dblp.org/search/publ/api"
        params = {'q': query, 'h': 1, 'format': 'json'}
        resp = requests.get(url, params=params)
        data = resp.json()
        
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
        if not hits:
            return None
            
        top_hit = hits[0]
        info = top_hit.get('info', {})
        found_title = info.get('title', "No Title")
        
        # Check match
        if is_match(query, found_title):
            print(f"✅ Found on DBLP: {found_title}", file=sys.stderr)
            # DBLP BibTeX endpoint: https://dblp.org/rec/{key}.bib
            key = info.get('key')
            bib_url = f"https://dblp.org/rec/{key}.bib"
            bib_resp = requests.get(bib_url)
            bib_resp.raise_for_status()  # Check for 429, 500, etc.
            return bib_resp.text
        else:
            print(f"⚠️  DBLP mismatch (Query: '{query}' vs Found: '{found_title}')", file=sys.stderr)
            return None
            
    except requests.exceptions.HTTPError as e:
        print(f"❌ DBLP HTTP Error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"❌ DBLP Error: {e}", file=sys.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Fetch BibTeX for a paper title using Crossref (primary) and DBLP (fallback).")
    parser.add_argument("paper_name", type=str, help="The full title of the paper.")
    args = parser.parse_args()

    query = args.paper_name
    print(f"🔎 Searching for: '{query}'...", file=sys.stderr)

    # 1. Try Crossref
    bibtex = get_bibtex_crossref(query)

    # 2. Try DBLP if Crossref fails
    if not bibtex:
        print("🔄 Switching to DBLP fallback...", file=sys.stderr)
        bibtex = get_bibtex_dblp(query)

    # 3. Output
    if bibtex:
        print("\n" + bibtex)
    else:
        print("\n❌ Could not find an exact match in Crossref or DBLP.", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()