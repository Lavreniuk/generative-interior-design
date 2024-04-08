"""
Extract photo urls and captions from photos.json files
"""
from pathlib import Path
from typing import Set, List, Dict
import csv
import json
import tap
from tqdm.auto import tqdm
from helpers import load_json  # type: ignore


class Arguments(tap.Tap):
    merlin: Path
    output: Path = Path("photos.csv")


if __name__ == "__main__":

    args = Arguments().parse_args()
    print(args)

    # Extract photos
    photos: List[Dict] = []
    listing_ids: Set[int] = set()
    counter = 0

    for filename in tqdm(args.merlin.rglob("photos.json")):
        listing_id = int(filename.parent.name)
        if listing_id in listing_ids:
            continue
        counter += 1
        try:
            data = load_json(filename)
        except json.decoder.JSONDecodeError:
            continue

        listing_ids.add(listing_id)

        if "data" not in data:
            continue

        photo_tour = data["data"]["merlin"]["pdpPhotoTour"]
        if photo_tour is None:
            continue

        for image in photo_tour["images"]:
            photos.append(
                {
                    "listing_id": listing_id,
                    "photo_id": image["id"],
                    "url": image["baseUrl"],
                    "caption": image["imageMetadata"]["caption"],
                }
            )
    print(f"Extracted {len(photos)} photos from {counter} listings")
    assert len(photos) > 0

    # Store photos
    with open(args.output, "w", newline="") as fid:
        fieldnames = photos[0].keys()
        writer = csv.DictWriter(fid, fieldnames=fieldnames, delimiter="\t")
        writer.writerows(photos)
