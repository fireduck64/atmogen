import os
import json
import shutil
from typing import Optional
from pathlib import Path

LIBRARY_DIR = "audio_library"
MANIFEST_FILE = os.path.join(LIBRARY_DIR, "manifest.json")

def init_library():
    """Ensures the audio library directory and manifest exist."""
    os.makedirs(LIBRARY_DIR, exist_ok=True)
    if not os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "w") as f:
            json.dump({}, f, indent=4)

def get_asset_path(slug: str, revision: int) -> Optional[str]:
    """Returns the path to a specific audio asset version."""
    filename = f"{slug}_v{revision}.mp3"
    path = os.path.join(LIBRARY_DIR, filename)
    if os.path.exists(path):
        return path
    return None

def resolve_asset_metadata(slug: str, revision: int) -> Optional[dict]:
    """Returns metadata for a specific audio asset version from the manifest."""
    if not os.path.exists(MANIFEST_FILE):
        return None
    
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    
    asset = manifest.get(slug)
    if asset and str(revision) in asset:
        return asset[str(revision)]
    
    # Handle case where revision is an integer key in JSON (which are always strings)
    if asset and str(revision) not in asset:
        # check if it exists as int just in case, though json leads to strings
        pass
        
    return None

def register_asset(slug: str, revision: int, file_path: str, metadata: dict):
    """Registers a new audio asset and updates the manifest."""
    init_library()
    
    with open(MANIFEST_FILE, "r") as f:
        manifest = json.load(f)
    
    if slug not in manifest:
        manifest[slug] = {}
    
    manifest[slug][str(revision)] = metadata
    
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=4)
