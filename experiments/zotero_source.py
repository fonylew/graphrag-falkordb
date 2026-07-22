"""Portable Zotero PDF discovery for experiment scripts.

Unlike ingest_zotero.py / manage_rag.py, this does not hardcode a user's
home directory or a specific collection ID. It auto-detects the Zotero data
directory and can pull PDF attachments from the whole library.
"""

import os
import random
import shutil
import sqlite3
import sys
import tempfile

# Query every PDF attachment in the library (no collection filter).
_ALL_PDFS_QUERY = """
SELECT
    attachments.key AS attachmentKey,
    itemAttachments.path
FROM itemAttachments
JOIN items AS attachments ON itemAttachments.itemID = attachments.itemID
WHERE itemAttachments.contentType = 'application/pdf';
"""


def find_zotero_root(search_root: str | None = None) -> str:
    """Locate the Zotero data directory (the one containing zotero.sqlite).

    Zotero's data directory is usually ~/Zotero, but on some installs it is
    nested one level deeper (~/Zotero/Zotero). Search a couple of likely
    spots instead of hardcoding either.
    """
    search_root = search_root or os.path.expanduser("~/Zotero")
    candidates = [
        search_root,
        os.path.join(search_root, "Zotero"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "zotero.sqlite")):
            return candidate

    for dirpath, _dirnames, filenames in os.walk(search_root):
        if "zotero.sqlite" in filenames:
            return dirpath

    raise FileNotFoundError(
        f"Could not find zotero.sqlite under {search_root}. "
        "Pass an explicit --zotero-root."
    )


def get_all_pdf_paths(zotero_root: str) -> list[str]:
    """Return absolute paths for every PDF attachment in the whole library."""
    db_path = os.path.join(zotero_root, "zotero.sqlite")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"zotero.sqlite not found at {db_path}")

    # Zotero locks its DB file while running; read from a temp copy.
    fd, copy_path = tempfile.mkstemp(suffix=".sqlite", prefix="zotero_copy_")
    os.close(fd)
    shutil.copy2(db_path, copy_path)

    try:
        conn = sqlite3.connect(copy_path)
        cursor = conn.cursor()
        cursor.execute(_ALL_PDFS_QUERY)
        rows = cursor.fetchall()
        print(f"Found {len(rows)} PDF attachment records in the Zotero library.", file=sys.stderr)

        pdf_paths = []
        for attach_key, rel_path in rows:
            if not rel_path or not attach_key:
                continue

            if rel_path.startswith("storage:"):
                filename = rel_path[len("storage:"):]
                full_path = os.path.join(zotero_root, "storage", attach_key, filename)
            else:
                full_path = rel_path

            if os.path.exists(full_path):
                pdf_paths.append(full_path)

        print(f"Verified {len(pdf_paths)} out of {len(rows)} files exist on disk.", file=sys.stderr)
        return pdf_paths
    finally:
        conn.close()
        if os.path.exists(copy_path):
            os.remove(copy_path)


def sample_paths(paths: list[str], n: int, seed: int = 42) -> list[str]:
    """Deterministically shuffle once and take the first n paths.

    Using one fixed shuffle order means the sample for a smaller n is always
    a prefix of the sample for a larger n, so batch-size comparisons aren't
    confounded by drawing a different mix of documents at each size.
    """
    shuffled = paths[:]
    random.Random(seed).shuffle(shuffled)
    return shuffled[:n]


if __name__ == "__main__":
    root = find_zotero_root()
    print(f"Zotero root: {root}")
    all_paths = get_all_pdf_paths(root)
    print(f"Total PDFs available: {len(all_paths)}")
    for p in sample_paths(all_paths, 5):
        print(f"  - {p}")
