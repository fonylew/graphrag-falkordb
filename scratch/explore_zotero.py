import sqlite3
import os
import shutil

db_path = "/home/fony/Zotero/zotero.sqlite"
copy_path = "/home/fony/graphrag-falkordb/zotero_copy.sqlite"

shutil.copy2(db_path, copy_path)
conn = sqlite3.connect(copy_path)
cursor = conn.cursor()

query = """
SELECT 
    items.key AS itemKey,
    attachments.key AS attachmentKey,
    itemAttachments.path,
    itemAttachments.contentType
FROM collectionItems
JOIN items ON collectionItems.itemID = items.itemID
LEFT JOIN itemAttachments ON itemAttachments.parentItemID = items.itemID
LEFT JOIN items AS attachments ON itemAttachments.itemID = attachments.itemID
WHERE collectionItems.collectionID = 37 AND itemAttachments.contentType = 'application/pdf';
"""

cursor.execute(query)
rows = cursor.fetchall()
print(f"Found {len(rows)} PDF attachments in the GraphRAG collection:")

found_count = 0
for i, row in enumerate(rows):
    item_key, attach_key, rel_path, content_type = row
    if rel_path.startswith("storage:"):
        filename = rel_path[len("storage:"):]
        full_path = f"/home/fony/Zotero/storage/{attach_key}/{filename}"
        exists = os.path.exists(full_path)
        if exists:
            found_count += 1
        print(f"{i+1}. {filename} -> {'EXISTS' if exists else 'NOT FOUND'} at {full_path}")
    else:
        print(f"{i+1}. Non-storage path: {rel_path}")

print(f"\nSuccessfully verified {found_count} out of {len(rows)} PDF files.")

conn.close()
if os.path.exists(copy_path):
    os.remove(copy_path)
