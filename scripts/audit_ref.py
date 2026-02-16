#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path

# --- Configuration ---
ROOT_DIR = Path(__file__).parent.parent
LEARNING_DIR = ROOT_DIR / "Learning"
BIB_FILE = ROOT_DIR / "references.bib"
INDEX_FILE = ROOT_DIR / "scripts" / "bib_index.json"

def clean_key(key):
    """Normalize key to be concise, ASCII, and underscored."""
    # Handle acronyms in parentheses like "PSRO (SP-PSRO)" -> "SP_PSRO"
    match = re.search(r"\(([^)]+)\)", key)
    if match:
        key = match.group(1)
    
    # Replace non-ascii (like α) with ascii equivalents or remove
    key = key.replace("α", "Alpha").replace("β", "Beta").replace("—", "-")
    
    # Remove everything after ":" or " - " (handles "Title: Subtitle")
    key = key.split(":")[0].split(" - ")[0]
    
    # Remove special chars, replace spaces/hyphens with underscore
    key = re.sub(r"[^\w\s-]", "", key)
    key = re.sub(r"[\s-]+", "_", key)
    
    # If still too long, take the first 3 words
    words = [w for w in key.split("_") if w]
    if len(key) > 25 and len(words) > 3:
        key = "_".join(words[:3])

    # Final truncation safeguard
    if len(key) > 30:
        key = key[:30].strip("_")
        
    return key.strip("_")

def audit_and_fix():
    if not INDEX_FILE.exists():
        print("No index file found.")
        return

    with open(INDEX_FILE, "r") as f:
        index = json.load(f)

    # 1. Collect all notes in Learning/
    all_notes = sorted([p for p in LEARNING_DIR.rglob("*.md")])
    
    # Map base names to their associated files (e.g., "X.md" -> ["X.md", "X_detailed.md"])
    note_groups = {}
    for p in all_notes:
        rel_path = str(p.relative_to(ROOT_DIR))
        base_path = rel_path.replace("_detailed.md", ".md")
        if base_path not in note_groups:
            note_groups[base_path] = []
        note_groups[base_path].append(rel_path)

    indexed_paths = set(index.keys())
    
    report = {
        "missing_paper_info": [],
        "index_or_moc": [],
        "fetch_failed": [],
        "indexed": []
    }

    for base_path, files in note_groups.items():
        # A group is "indexed" if ANY of its files are in bib_index.json
        is_indexed = any(f in indexed_paths for f in files)
        
        if is_indexed:
            report["indexed"].extend(files)
            continue
            
        # If not indexed, check the primary file (or all if necessary) for why
        # Just use the first one to determine the category for the group
        p = ROOT_DIR / files[0]
        
        # Determine why missing
        if p.name.lower() in ["readme.md", "index.md", "learning.md", "game theory.md", "machine learning.md", "evolutionary optimization.md", "reinforcement learning.md"] or "MOC" in p.name:
            report["index_or_moc"].extend(files)
            continue

        try:
            with open(p, "r") as f:
                content = "".join([f.readline() for _ in range(20)])
                has_title = any(re.search(pat, content, re.I) for pat in [r"\*\*Title\*\*", r"\*\*Paper\*\*", r"Paper:"])
                
                if not has_title:
                    report["missing_paper_info"].extend(files)
                else:
                    report["fetch_failed"].extend(files)
        except Exception:
            report["fetch_failed"].extend(files)

    print(f"\n=== AUDIT REPORT ===")
    print(f"Total notes scanned: {len(all_notes)}")
    print(f"Already Indexed: {len(report['indexed'])}")
    
    def print_consolidated(paths):
        # Group paths by base name to avoid listing _detailed separately in report
        groups = {}
        for p in paths:
            base = p.replace("_detailed.md", ".md")
            if base not in groups: groups[base] = []
            groups[base].append(p)
        
        for base in sorted(groups.keys()):
            files = groups[base]
            if len(files) > 1:
                print(f"- {base} (+ _detailed)")
            else:
                print(f"- {files[0]}")

    print(f"\n--- Missing Paper Info ({len(report['missing_paper_info'])}) ---")
    print_consolidated(report['missing_paper_info'])
        
    print(f"\n--- Fetch Failed / Needs Retry ({len(report['fetch_failed'])}) ---")
    print_consolidated(report['fetch_failed'])

    print(f"\n--- Index/MOC/Overview (Ignored) ({len(report['index_or_moc'])}) ---")

    # 2. Fix keys in index and references.bib
    new_index = {}
    key_mapping = {} 

    for path, entry in index.items():
        # Prefer algo_name for generating the key if it looks better than the current messy key
        new_key = clean_key(entry.get("algo_name", entry["bib_key"]))
        
        old_key = entry["bib_key"]
        entry["bib_key"] = new_key
        new_index[path] = entry
        key_mapping[old_key] = new_key

    # Update references.bib
    if BIB_FILE.exists():
        with open(BIB_FILE, "r") as f:
            bib_content = f.read()

        fixed_bib = bib_content
        # Sort by length descending to avoid partial replacements
        for old_key in sorted(key_mapping.keys(), key=len, reverse=True):
            new_key = key_mapping[old_key]
            if old_key != new_key:
                # Use regex to find @type{old_key, and replace with new_key
                pattern = rf"(@\w+\{{){re.escape(old_key)}(?=,)"
                fixed_bib = re.sub(pattern, rf"\1{new_key}", fixed_bib)

        with open(BIB_FILE, "w") as f:
            f.write(fixed_bib)

    with open(INDEX_FILE, "w") as f:
        json.dump(new_index, f, indent=4)

    print("\n--- Key Normalization Complete ---")
    print(f"Fixed {sum(1 for k, v in key_mapping.items() if k != v)} keys.")


if __name__ == "__main__":
    audit_and_fix()
