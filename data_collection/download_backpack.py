import requests
from bs4 import BeautifulSoup
import os
import hashlib
import time
from PIL import Image
from io import BytesIO
import json

print("=" * 70)
print("BACKPACK IMAGE DOWNLOADER - 2000 IMAGES")
print("Bulletproof with gap filling and duplicate detection")
print("=" * 70)

TARGET_IMAGES = 2000
OUTPUT_FOLDER = "backpack_dataset_2000"
STATE_FILE = "download_state_backpack.json"
MIN_IMAGE_SIZE = (200, 200)

SEARCH_QUERIES = [
    "backpack school",
    "backpack student",
    "backpack college",
    "backpack hiking",
    "backpack travel",
    "backpack outdoor",
    "backpack isolated",
    "backpack white background",
    "backpack black background",
    "backpack close up",
    "backpack front view",
    "backpack side view",
    "backpack product photo",
    "laptop backpack",
    "rucksack backpack",
    "backpack camping",
    "backpack adventure",
    "backpack brand",
    "backpack empty",
    "backpack zipper",
    "backpack straps",
    "backpack compartments",
    "backpack design",
    "backpack modern",
    "backpack professional",
    "backpack business",
    "backpack work",
    "backpack urban",
    "backpack fashion",
    "backpack sport",
    "backpack waterproof",
    "backpack leather",
    "backpack nylon",
    "backpack canvas",
    "backpack studio shot"
]


def get_image_hash(image_data):
    return hashlib.md5(image_data).hexdigest()


def scan_existing_images():
    print("\nScanning existing backpack images...")
    existing_indices = set()
    existing_hashes = {}

    if os.path.exists(OUTPUT_FOLDER):
        for filename in os.listdir(OUTPUT_FOLDER):
            if filename.startswith('backpack_') and filename.endswith('.jpg'):
                try:
                    index = int(filename.split('_')[1].split('.')[0])
                    existing_indices.add(index)

                    img_path = os.path.join(OUTPUT_FOLDER, filename)
                    with open(img_path, 'rb') as f:
                        img_hash = hashlib.md5(f.read()).hexdigest()
                        existing_hashes[img_hash] = index
                except:
                    continue

    if existing_indices:
        max_index = max(existing_indices)
        all_indices = set(range(max_index + 1))
        missing_indices = sorted(all_indices - existing_indices)
    else:
        max_index = -1
        missing_indices = []

    return existing_indices, existing_hashes, missing_indices, max_index


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {'last_query_index': 0, 'failed_urls': []}


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)


def is_valid_image(image_data):
    try:
        img = Image.open(BytesIO(image_data))
        width, height = img.size
        if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
            return False
        img.convert('RGB')
        return True
    except:
        return False


def download_from_bing(query, num_images=100):
    image_urls = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    for offset in range(0, num_images, 35):
        try:
            url = f"https://www.bing.com/images/search?q={query}&first={offset}"
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for img in soup.find_all('a', class_='iusc'):
                    if 'm' in img.attrs:
                        import json as json_module
                        try:
                            m = json_module.loads(img['m'])
                            if 'murl' in m:
                                image_urls.append(m['murl'])
                        except:
                            pass
            time.sleep(0.5)
        except:
            continue
    return image_urls


def download_image(url, timeout=10):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    existing_indices, existing_hashes, missing_indices, max_index = scan_existing_images()

    print(f"\nCURRENT DATASET STATUS:")
    print(f"   Existing images: {len(existing_indices)}")
    print(f"   Highest index: {max_index}")
    print(f"   Missing gaps: {len(missing_indices)}")
    print(f"   Target: {TARGET_IMAGES}")
    print(f"   Need: {TARGET_IMAGES - len(existing_indices)} more")

    if len(existing_indices) >= TARGET_IMAGES:
        print("\nALREADY HAVE 2000 BACKPACK IMAGES!")
        return

    state = load_state()
    print("\n" + "=" * 70)
    print("STARTING DOWNLOAD PROCESS")
    print("=" * 70)

    target_indices = missing_indices.copy()
    target_indices.extend(range(max_index + 1, TARGET_IMAGES))
    print(f"\nTarget indices to fill: {len(target_indices)}")

    query_index = state['last_query_index']
    current_target_idx = 0

    while current_target_idx < len(target_indices):
        if query_index >= len(SEARCH_QUERIES):
            query_index = 0
            print("\nCycling through queries...")

        query = SEARCH_QUERIES[query_index]
        print(f"\n[Query {query_index + 1}/{len(SEARCH_QUERIES)}] '{query}'")
        print("-" * 70)

        image_urls = download_from_bing(query, num_images=150)
        print(f"   Found {len(image_urls)} potential URLs")

        downloaded_this_query = 0
        skipped_duplicate = 0
        skipped_invalid = 0

        for url_idx, url in enumerate(image_urls):
            if current_target_idx >= len(target_indices):
                break
            if url in state['failed_urls']:
                continue
            if (url_idx + 1) % 30 == 0:
                print(f"      Progress: {url_idx + 1}/{len(image_urls)} | Downloaded: {downloaded_this_query}")

            image_data = download_image(url)
            if image_data is None:
                state['failed_urls'].append(url)
                continue

            img_hash = get_image_hash(image_data)
            if img_hash in existing_hashes:
                skipped_duplicate += 1
                continue

            if not is_valid_image(image_data):
                skipped_invalid += 1
                state['failed_urls'].append(url)
                continue

            target_index = target_indices[current_target_idx]
            img_path = os.path.join(OUTPUT_FOLDER, f"backpack_{target_index:04d}.jpg")

            try:
                img = Image.open(BytesIO(image_data))
                img = img.convert('RGB')
                img.save(img_path, 'JPEG', quality=95)
                existing_hashes[img_hash] = target_index
                current_target_idx += 1
                downloaded_this_query += 1
                if downloaded_this_query % 5 == 0:
                    save_state(state)
            except:
                skipped_invalid += 1
                state['failed_urls'].append(url)
                continue

        remaining = len(target_indices) - current_target_idx
        total_now = len(existing_indices) + current_target_idx

        print(f"\n   Query Results:")
        print(f"      Downloaded: {downloaded_this_query}")
        print(f"      Duplicates: {skipped_duplicate}")
        print(f"      Invalid: {skipped_invalid}")
        print(f"      Total Progress: {total_now}/{TARGET_IMAGES}")
        print(f"      Remaining: {remaining}")

        query_index += 1
        state['last_query_index'] = query_index
        save_state(state)

        if remaining > 0:
            time.sleep(3)

    print("\n" + "=" * 70)
    print("FINAL VERIFICATION")
    print("=" * 70)

    final_indices, _, final_missing, _ = scan_existing_images()
    print(f"   Total images: {len(final_indices)}")
    print(f"   Missing gaps: {len(final_missing)}")

    print("\nDOWNLOAD COMPLETE!")
    print(f"   Images: {len(final_indices)}/{TARGET_IMAGES}")
    print(f"   Location: {os.path.abspath(OUTPUT_FOLDER)}")

    if len(final_indices) >= TARGET_IMAGES:
        print("   SUCCESS! 2000 backpack images complete!")
    else:
        print(f"   Re-run to continue (need {TARGET_IMAGES - len(final_indices)} more)")

    save_state(state)


try:
    main()
except KeyboardInterrupt:
    print("\n\nINTERRUPTED - Progress saved")
except Exception as e:
    print(f"\n\nERROR: {e}")
    print("Progress saved")