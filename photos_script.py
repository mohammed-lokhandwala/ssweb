import requests
import os
import time

# 🔑 Your Access Key (use ONLY access key, not secret)
ACCESS_KEY = "jUrB9w4A6Gk9PT2M1liTHX3e9jUiO3DFPoPHHbz7T5I"

# 📁 Folder setup
DOWNLOAD_DIR = "wedding_photos"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ⚙️ Config
TOTAL_IMAGES = 200
PER_PAGE = 30
QUERY = "wedding guests bride groom event"

downloaded = 0
page = 1

while downloaded < TOTAL_IMAGES:
    print(f"Fetching page {page}...")

    url = "https://api.unsplash.com/search/photos"
    params = {
        "query": QUERY,
        "per_page": PER_PAGE,
        "page": page,
        "client_id": ACCESS_KEY
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("Error:", response.text)
        break

    data = response.json()
    results = data.get("results", [])

    if not results:
        print("No more images found.")
        break

    for img in results:
        if downloaded >= TOTAL_IMAGES:
            break

        try:
            img_url = img["urls"]["regular"]  # good balance quality/size
            img_id = img["id"]

            img_data = requests.get(img_url).content

            file_path = os.path.join(DOWNLOAD_DIR, f"{img_id}.jpg")
            with open(file_path, "wb") as f:
                f.write(img_data)

            downloaded += 1
            print(f"Downloaded {downloaded}/{TOTAL_IMAGES}")

            time.sleep(0.2)  # avoid hammering API

        except Exception as e:
            print("Skip image:", e)

    page += 1

print("✅ Done. Total downloaded:", downloaded)