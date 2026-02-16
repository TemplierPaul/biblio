#!/usr/bin/env python3
import os
import re
import json
import subprocess
import time
from pathlib import Path


# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent
LEARNING_DIR = ROOT_DIR / "Learning"
BIB_FILE = ROOT_DIR / "references.bib"
INDEX_FILE = ROOT_DIR / "scripts" / "bib_index.json"
FETCH_BIB_SCRIPT = ROOT_DIR / "scripts" / "fetch_bib.py"
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python3"

# Extraction Regexes (First 10 lines)
TITLE_PATTERNS = [
    re.compile(r"\*\*Title\*\*:\s*(.*)", re.IGNORECASE),
    re.compile(r"\*\*Paper\*\*:\s*(.*)", re.IGNORECASE),
    re.compile(r"\*\*Reference\*\*:\s*(.*)", re.IGNORECASE),
    re.compile(r"Paper:\s*(.*)", re.IGNORECASE),
]
ALGO_PATTERN = re.compile(r"^#\s+(.*)", re.MULTILINE)

def clean_algo_name(name):
    """Remove common note-taking suffixes from algorithm names."""
    suffixes = [
        r"\s*[:—\-–]\s*Detailed Implementation.*",
        r"\s*Detailed Implementation.*",
        r"\s*Implementation Notes.*",
        r"\s*Implementation Details.*",
        r"\s*Detailed Analysis.*",
        r"\s*Detailed Implementation Guide.*",
        r"\s*Detailed.*",
        r"\s*Notes.*",
        r"\s*\(Detailed\).*"
    ]
    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE).strip()
    return name

def get_note_info(file_path):
    """Parse first 10 lines of a note for algo name and paper title."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [f.readline() for _ in range(10)]
            content = "".join(lines)
            
            # 1. Extract Algo Name (First # header)
            algo_match = ALGO_PATTERN.search(content)
            raw_algo_name = algo_match.group(1).strip() if algo_match else file_path.stem
            algo_name = clean_algo_name(raw_algo_name)
            
            # 2. Extract Paper Title
            paper_title = None
            for pattern in TITLE_PATTERNS:
                match = pattern.search(content)
                if match:
                    paper_title = match.group(1).strip().strip('"').strip("'")
                    break
            
            return algo_name, paper_title
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None

def update_bibtex_key(bibtex, new_key):
    """Replace the first BibTeX key found with new_key."""
    return re.sub(r"(@\w+\{)([^,]+),", rf"\1{new_key},", bibtex, count=1)

def main():
    if not INDEX_FILE.exists():
        index = {}
    else:
        with open(INDEX_FILE, "r") as f:
            index = json.load(f)

    # Track existing keys in references.bib to avoid duplicates
    existing_keys = set()
    if BIB_FILE.exists():
        with open(BIB_FILE, "r") as f:
            content = f.read()
            existing_keys = set(re.findall(r"@\w+\{([^,]+),", content))

    notes = list(LEARNING_DIR.rglob("*.md"))
    updated = False

    for note_path in notes:
        if note_path.name.lower() in ["readme.md", "index.md"]:
            continue
            
        rel_path = str(note_path.relative_to(ROOT_DIR))
        algo_name, paper_title = get_note_info(note_path)
        
        if not paper_title:
            continue
            
        # Sanitize algo_name for BibTeX key
        safe_key = re.sub(r"\W+", "_", algo_name).strip("_")

        # Check if already indexed correctly
        if rel_path in index:
            entry = index[rel_path]
            if entry.get("paper_title") == paper_title and entry.get("bib_key") == safe_key:
                continue
            
        print(f"Processing: {rel_path} -> {paper_title} (Algo: {algo_name})")
        
        # If this key already exists in references.bib, we just link it in the index
        if safe_key in existing_keys:
            print(f"🔗 Linking {rel_path} to existing key: {safe_key}")
            index[rel_path] = {
                "algo_name": algo_name,
                "paper_title": paper_title,
                "bib_key": safe_key
            }
            updated = True
            continue

        # Fetch BibTeX if not already in references.bib
        try:
            result = subprocess.run(
                [str(PYTHON_BIN), str(FETCH_BIB_SCRIPT), paper_title],
                capture_output=True, text=True, check=True
            )
            bibtex = result.stdout.strip()
            
            if bibtex and bibtex.strip().startswith("@"):
                updated_bib = update_bibtex_key(bibtex, safe_key)
                
                with open(BIB_FILE, "a") as f:
                    f.write("\n" + updated_bib + "\n")
                
                existing_keys.add(safe_key)
                index[rel_path] = {
                    "algo_name": algo_name,
                    "paper_title": paper_title,
                    "bib_key": safe_key
                }
                updated = True
                print(f"✅ Added {safe_key} to references.bib")
                
                # Small delay to avoid rate limiting
                time.sleep(2.0)

            else:
                if bibtex:
                    print(f"⚠️ Invalid BibTeX received (possibly HTML error) for: {paper_title}")
                else:
                    print(f"⚠️ No BibTeX found for: {paper_title}")

                
        except subprocess.CalledProcessError as e:
            print(f"❌ Error fetching BibTeX for {paper_title}: {e.stderr}")

    if updated:
        with open(INDEX_FILE, "w") as f:
            json.dump(index, f, indent=4)
        print("\nSync complete. scripts/bib_index.json updated.")
    else:
        print("\nNo new papers found to sync.")


if __name__ == "__main__":
    main()
