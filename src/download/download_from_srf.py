import base64
import json
import os
import time

import requests

from src.download.utils import save_podcast_metadata_to_csv, load_podcast_metadata_from_csv, \
    create_audio_folder_if_not_exists, PODCAST_METADATA_FOLDER, PODCAST_AUDIO_FOLDER, get_downloaded_metadata
from src.utils.logger import get_logger

CONSUMER_KEY = "YOUR_CONSUMER_KEY"
CONSUMER_SECRET = "YOUR_CONSUMER_KEY"
AUTH_TOKEN = base64.b64encode(f"{CONSUMER_KEY}:{CONSUMER_SECRET}".encode()).decode()

URL_BASE = "https://api.srgssr.ch"
URL_CLIENT_CREDENTIALS = f"{URL_BASE}/oauth/v1/accesstoken?grant_type=client_credentials"
URL_AUDIOS = f"{URL_BASE}/audiometadata/v2"

logger = get_logger(__name__)

new_podcasts = [
    "Besserwisser",
    "Blick in die Feuilletons",
    "BuchZeichen",
    "Die verflixte Gebrauchsanweisung",
    "Echo der Zeit",
    "Einfach Politik",
    "Es geschah am... Postraub des Jahrhunderts",
    "Espresso",
    "Focus",
    "Forum",
    "Grauen",
    "Input",
    "Kontext",
    "Krimi",
    "Kultur kompakt",
    "News Plus",
    "Persönlich",
    "Perspektiven",
    "Politikum",
    "Ratgeber",
    "Rehmann",
    "Trend",
    "Trüffelschweine",
    "Wetter",
    "WortSchatz",
    "Zeitblende"
]

existing_podcasts = [  # 17278872.0 seconds
    "Debriefing 404",
    "Digital Podcast",
    "Dini Mundart",
    # "Dini Mundart Schnabelweid",  # <- no downloads available
    "Gast am Mittag",
    "Geek-Sofa",
    # "Giigets Die SRF 3-Alltagsphilosophie", # <- no downloads available
    # "Morgengast",  # <- no downloads available
    "Pipifax",
    "Podcast am Pistenrand",
    "Samstagsrundschau",
    # "Schwiiz und dütlich",  # <-- no downloads available
    "#SRFglobal",
    "Sykora Gisler",
    "Tagesgespräch",
    "Ufwärmrundi",
    "Vetters Töne",
    "Wetterfrage",
    "Zivadiliring",
    "Zytlupe",
    ############# from here its de only #####################
    "100 Sekunden Wissen",
    # "Kontext",  # can contain background noise
    "Kultur-Talk",
    "Kopf voran",
    "Literaturclub: Zwei mit Buch",
    "Medientalk",
    "Sternstunde Philosophie",
    "Sternstunde Religion",
    "Wirtschaftswoche",
    "Wissenschaftsmagazin"  # mixed (EN, DE, CHDE), can contain background noise
]


def _check_and_load_response(response: requests.Response) -> dict:
    if response.status_code in [200, 203]:
        return json.loads(response.text)
    else:
        raise RuntimeError(f"Failed to get response. Response code {response.status_code} with message {response.text}")


def get_access_token() -> dict:
    headers = {
        "Authorization": "Basic " + AUTH_TOKEN,
        "Cache-Control": "no-cache",
        "Content-Length": "0",
    }
    response = requests.post(URL_CLIENT_CREDENTIALS, headers=headers)
    return _check_and_load_response(response)


def _collect_metadata(media: list, current_podcast) -> list:
    episodes = []
    for episode in media:
        if episode["show"]["title"].lower() == current_podcast.lower():
            episodes.append({
                "id": episode["id"],
                "title": episode["title"],
                "description": episode.get("description", "NO_DESCRIPTION"),
                "date_published": episode.get("date", "NO_DATE"),
                "duration_s": episode["duration"] / 1000,
                "download_available": episode["downloadAvailable"],
                "subtitles_available": episode["subtitlesAvailable"],
                "url": episode.get("podcastHdUrl", "NO_URL"),
            })
    return episodes


def process_srf_podcast(podcast: str) -> None:
    download_srf_podcast_metadata(podcast)
    download_srf_podcast_audio(podcast)


def download_srf_podcast_metadata(podcast: str, skip: bool = True) -> None:
    os.makedirs(PODCAST_METADATA_FOLDER, exist_ok=True)
    url = URL_AUDIOS + "/audios/search"

    access_token = get_access_token()
    headers = {
        "Authorization": f"{access_token['token_type']} {access_token['access_token']}",
        "Cache-Control": "no-cache",
        "accept": "application/json"
    }

    saved_podcasts = get_downloaded_metadata()

    if skip and f"{podcast}.csv" in saved_podcasts:
        logger.warning(f"Podcast {podcast} already downloaded")
        return

    params = {
        "bu": "srf",
        "q": podcast,
        "pageSize": 100
    }

    response = requests.get(url, headers=headers, params=params)
    json_response = _check_and_load_response(response)

    episodes = []
    total_episodes = json_response["total"]
    logger.info(f"Getting podcast {podcast} with total number of episodes: {json_response['total']}")
    episodes.extend(_collect_metadata(json_response["searchResultListMedia"], podcast))

    while "next" in json_response:
        params["next"] = json_response["next"].split("?")[1].replace("next=", "").split("&")[0]
        try:
            response = requests.get(url, headers=headers, params=params)
            json_response = _check_and_load_response(response)
        except Exception as e:
            logger.error(str(e))
            break
        episodes.extend(_collect_metadata(json_response["searchResultListMedia"], podcast))

    save_podcast_metadata_to_csv(podcast, episodes)
    logger.info(f"Expected number of podcasts: {total_episodes}, saved {len(episodes)}")


def download_srf_podcast_audio(podcast: str) -> None:
    df = load_podcast_metadata_from_csv(podcast)
    create_audio_folder_if_not_exists(podcast)

    for i, metadata in df.iterrows():
        ep_path = f"{PODCAST_AUDIO_FOLDER}/{podcast}/{metadata['id']}.mp3"
        if not metadata["download_available"] or os.path.exists(ep_path):
            continue

        response = requests.get(metadata["url"], allow_redirects=True)
        with open(ep_path, 'wb') as f:
            f.write(response.content)

        logger.info(f"downloaded {metadata['id']} for {podcast}")
        time.sleep(0.25)
