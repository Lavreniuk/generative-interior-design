from typing import List, Union, Dict, Iterable, Sequence
import csv
from pathlib import Path
import hashlib
import json
import requests


class DownloadError(Exception):
    """Unknown error during download"""


def download_file(url: str, dest: Union[Path, str]):
    if Path(dest).is_file():
        return
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise DownloadError()
    with open(dest, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)


class TooManyRequestsError(Exception):
    """Can't download an image"""


def get_slug(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()


def graphql_request(key: str, params: Dict, proxy=None) -> Dict:
    url = "https://www.airbnb.com/api/v3/" + key
    proxies = None if proxy is None else {"https": proxy}
    r = requests.get(
        url,
        params=params,
        headers={
            "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:81.0) Gecko/20100101 Firefox/81.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "Content-Type": "application/json",
            "X-Airbnb-GraphQL-Platform": "web",
            "X-Airbnb-API-Key": "d306zoyjsyarp7ifhu67rjxn52tv0t20",
            "X-CSRF-Without-Token": "1",
            "X-Airbnb-GraphQL-Platform-Client": "minimalist-niobe",
        },
        proxies=proxies,
    )

    if r.status_code == 429:
        raise TooManyRequestsError(f"URL: {url}")
    elif r.status_code != 200:
        raise DownloadError(f"Status: {r.status_code}, URL: {url}")

    content = json.loads(r.text)

    if "errors" in content:
        raise DownloadError(content["errors"][0]["message"])

    return content


def load_txt(filename: Union[Path, str]):
    with open(filename) as fid:
        return [l.strip() for l in fid.readlines()]


def save_txt(data: Iterable, filename: Union[Path, str]):
    with open(filename, "w") as fid:
        for item in data:
            print(item, file=fid)


def get_key(listing_id: Union[str, int], photo_id: Union[str, int]) -> str:
    return f"{listing_id}-{photo_id}"


def location_to_slug(location: str):
    return location.replace(" ", "-").replace(",", "").lower()


def load_json(filename: Union[str, Path]):
    with open(filename, "r") as fid:
        return json.load(fid)


def save_json(data, filename: Union[str, Path]):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=2)


def is_downloaded(folder: Path):
    return (folder / "photos.json").is_file()


def is_empty(stc: str) -> bool:
    return stc.strip("., \n") == ""


def load_tsv(filename: Union[str, Path], fieldnames: Sequence[str]) -> List[Dict]:
    with open(filename, newline="") as fid:
        reader = csv.DictReader(fid, fieldnames=fieldnames, delimiter="\t")
        return list(reader)


def save_tsv(rows: Sequence[Dict], filename: Union[str, Path], fieldnames=None):
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(filename, "w", newline="") as fid:
        writer = csv.DictWriter(
            fid, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore"
        )
        writer.writerows(rows)
