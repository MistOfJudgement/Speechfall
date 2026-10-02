import io
import json
import os
import time
import urllib.parse
import webbrowser
import wave
from faster_whisper import WhisperModel
import numpy as np
import pyaudio
from rapidfuzz import fuzz, process, utils
import requests

SET_CODE = "fra"  # Reality Fracture set code
CACHE_FILE = f"mtg_set_{SET_CODE}_cards.json"
DECK_FILE = "deck.txt"
HEADERS = {
    "User-Agent": "MTGVoiceLookup/1.0",
    "Accept": "application/json",
}
MODEL_SIZE = "base"

# Words that trigger an undo action
UNDO_TRIGGERS = {"backspace", "undo", "scratch that"}


def get_set_card_database():
  """Downloads only the cards from the Reality Fracture set."""
  if os.path.exists(CACHE_FILE):
    print(f"Loading local cache for set [{SET_CODE.upper()}].")
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
      return json.load(f)

  print(
      f"First-time setup: Fetching card list for set '{SET_CODE.upper()}' from"
      " Scryfall..."
  )
  url = f"https://api.scryfall.com/cards/search?q=set:{SET_CODE}"

  card_names = set()
  while url:
    resp = requests.get(url, headers=HEADERS).json()
    for card in resp.get("data", []):
      name = card.get("name")
      if name:
        card_names.add(name)

    if resp.get("has_more"):
      url = resp.get("next_page")
    else:
      url = None

  card_names_list = list(card_names)
  with open(CACHE_FILE, "w", encoding="utf-8") as f:
    json.dump(card_names_list, f)

  print(
      f"Cached {len(card_names_list)} cards for Reality Fracture successfully!"
  )
  return card_names_list


def save_deck(deck_list):
  """Writes the current deck list to disk in '1 CARDNAME' format."""
  with open(DECK_FILE, "w", encoding="utf-8") as f:
    for card in deck_list:
      f.write(f"1 {card}\n")


def main():
  cards = get_set_card_database()

  # Create an initial prompt vocabulary list for Whisper to heavily bias set card names
  whisper_vocab_prompt = (
      f"Magic: The Gathering card names from Reality Fracture: "
      + ", ".join(cards[:300])
  )

  print(
      f"Loading local Whisper model ('{MODEL_SIZE}') into memory (runs entirely"
      " offline)..."
  )
  model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

  p = pyaudio.PyAudio()
  RATE = 16000
  CHUNK = 1024
  FORMAT = pyaudio.paInt16
  CHANNELS = 1

  stream = p.open(
      format=FORMAT,
      channels=CHANNELS,
      rate=RATE,
      input=True,
      frames_per_buffer=CHUNK,
  )

  session_deck = []

  print(
      f"\n>>> READY (Reality Fracture Set Scope + Local Whisper)!"
      f"\n    - Speak a card name fully (4s window)."
      f"\n    - Say 'backspace', 'undo', or 'scratch' to remove the last card."
      f"\n    - Press Ctrl+C to quit.\n"
  )

  try:
    while True:
      frames = []
      for _ in range(0, int(RATE / CHUNK * 3.0)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)

      audio_data = b"".join(frames)

      # --- NOISE / SILENCE GATE ---
      # Convert raw bytes to numpy array to measure volume (RMS)
      audio_np_raw = np.frombuffer(audio_data, dtype=np.int16)
      rms = np.sqrt(np.mean(audio_np_raw.astype(np.float32) ** 2))

      # If it's just quiet room noise, skip Whisper entirely!
      # (Adjust 300.0 up if your mic is sensitive, down if it misses soft voices)
      if rms < 300.0:
        continue
      # ----------------------------

      start_time = time.time()
      audio_np = audio_np_raw.astype(np.float32) / 32768.0

      segments, _ = model.transcribe(
          audio_np,
          beam_size=5,
          language="en",
          without_timestamps=True,
          initial_prompt=whisper_vocab_prompt,
      )
      spoken_text = " ".join([segment.text for segment in segments]).strip()
      spoken_text_clean = spoken_text.strip(".,!? ").lower()

      if not spoken_text_clean:
        continue

      # --- ROBUST COMMAND CHECK ---
      undo_keywords = {"backspace", "undo", "scratch", "thratch", "cancel"}
      is_undo = any(
          keyword in spoken_text_clean for keyword in undo_keywords
      ) or spoken_text_clean.startswith("scratch")

      if is_undo:
        if session_deck:
          removed = session_deck.pop()
          save_deck(session_deck)
          print(
              f"\n[Undo/Backspace]: Removed '{removed}' from {DECK_FILE}."
              f" ({len(session_deck)} cards left)"
          )
        else:
          print("\n[Undo/Backspace]: Deck list is already empty.")
        continue
      # ----------------------------

      # Fuzzy match against Reality Fracture cards
      match, score, _ = process.extractOne(
          spoken_text,
          cards,
          scorer=fuzz.token_set_ratio,
          processor=utils.default_process,
      )

      elapsed = (time.time() - start_time) * 1000
      print(
          f"[Spoken]: '{spoken_text}' -> [Match]: '{match}' (Score:"
          f" {score:.1f}) in {elapsed:.1f}ms"
      )

      if score > 50:
        session_deck.append(match)
        save_deck(session_deck)
        print(f"[Logged]: Added to {DECK_FILE} total: {len(session_deck)}")

        encoded_match = urllib.parse.quote(match)
        scryfall_url = (
            f'https://scryfall.com/search?q=set:{SET_CODE}+!"{encoded_match}"'
        )
        # webbrowser.open(scryfall_url)
      else:
        print("[!] Low confidence match, ignoring.")
  except KeyboardInterrupt:
    print(f"\nShutting down. Final deck saved to {DECK_FILE}.")
  finally:
    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
  main()