import gzip
import io
import json
import os
import time
import webbrowser
from rapidfuzz import process, fuzz, utils
import requests
import speech_recognition as sr

CACHE_FILE = "mtg_card_names.json"
HEADERS = {
    "User-Agent": "MTGVoiceLookup/1.0",
    "Accept": "application/json",
}

def get_card_database():
    """Downloads Scryfall's jsonl.gz bulk data, parses names, and caches them locally."""
    if os.path.exists(CACHE_FILE):
        print("Loading local card database into memory...")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    print("First-time setup: Fetching bulk data metadata from Scryfall...")
    meta_resp = requests.get("https://api.scryfall.com/bulk-data/default-cards", headers=HEADERS).json()
    
    # Scryfall supplies jsonl_download_uri for the compressed jsonl file
    download_url = meta_resp.get("jsonl_download_uri")
    print(f"Downloading compressed card data (~75MB)...")
    
    stream_resp = requests.get(download_url, headers=HEADERS, stream=True)
    
    card_names = set()
    print("Decompressing and extracting card names...")
    
    # Read the gzipped content chunk-by-chunk and unpack line by line
    with gzip.GzipFile(fileobj=io.BytesIO(stream_resp.content)) as gz:
        for line in gz:
            if line.strip():
                card_obj = json.loads(line.decode("utf-8"))
                name = card_obj.get("name")
                if name:
                    card_names.add(name)

    card_names_list = list(card_names)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(card_names_list, f)

    print(f"Cached {len(card_names_list)} unique cards successfully!")
    return card_names_list

def main():
    cards = get_card_database()
    
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True

    with sr.Microphone() as source:
        print("\nCalibrating microphone for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("\n>>> READY! Say a Magic card name (Ctrl+C to quit)...")

        while True:
            try:
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
                start_time = time.time()

                # Convert speech to text
                spoken_text = recognizer.recognize_google(audio)
                print(f"\n[Spoken]: '{spoken_text}'")

                # Local fuzzy match using RapidFuzz
                match, score, _ = process.extractOne(
                    spoken_text, 
                    cards, 
                    scorer=fuzz.WRatio,
                    processor=utils.default_process
                )

                elapsed = (time.time() - start_time) * 1000
                print(f"[Match]: '{match}' (Score: {score:.1f}) in {elapsed:.1f}ms")

                if score > 65:
                    scryfall_url = f"https://scryfall.com/search?q=!\"{requests.utils.quote(match)}\""
                    webbrowser.open(scryfall_url)
                else:
                    print("[!] Low confidence match. Try again.")

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                print("[!] Could not understand audio.")
            except Exception as e:
                print(f"[Error]: {e}")

if __name__ == "__main__":
    main()