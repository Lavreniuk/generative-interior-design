import csv
import json
import operator
import re
import time
import tap
import pandas as pd
import requests
from pathlib import Path
from typing import Union, Dict, List
from tqdm import tqdm

url = "https://www.airbnb.com/api/v3/StaysSearch?operationName=StaysSearch&locale=en&currency=USD"

headers = {
    "Origin": "https://www.airbnb.com",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Content-Type": "application/json",
    "X-Airbnb-API-Key": "d306zoyjsyarp7ifhu67rjxn52tv0t20",
}


class Arguments(tap.Tap):
    locations: Path  # txt files containing one location per line
    output: Path = Path("results_listing")


def get_request_body(location: str, price_min: int):
    return {'operationName': 'StaysSearch', 'variables': {
        'staysSearchRequest': {'requestedPageType': 'STAYS_SEARCH', 'metadataOnly': False, 'searchType': 'AUTOSUGGEST',
                               'rawParams': [
                                             {'filterName': 'adults', 'filterValues': ['1']},
                                             {'filterName': 'cdnCacheSafe', 'filterValues': ['false']},
                                             {'filterName': 'datePickerType', 'filterValues': ['flexible_dates']},
                                             {'filterName': 'flexibleTripDates',
                                              'filterValues': ['january', 'february', 'march', 'april', 'may', 'june',
                                                               'july', 'august', 'september', 'october', 'november',
                                                               'december']},
                                             {'filterName': 'flexibleTripLengths', 'filterValues': ['one_week']},
                                             # {'filterName': 'infants', 'filterValues': ['0']},
                                             {'filterName': 'itemsPerGrid', 'filterValues': ['500']},
                                             {'filterName': 'query', 'filterValues': [location]},
                                             {'filterName': 'refinementPaths', 'filterValues': ['/homes']},
                                             {'filterName': 'screenSize', 'filterValues': ['large']},
                                             {'filterName': 'tabId', 'filterValues': ['home_tab']},
                                             {'filterName': 'version', 'filterValues': ['1.8.3']},
                                             {'filterName': 'price_min', 'filterValues': [f"{price_min}"]},
                                             ], 'maxMapItems': 99999},
        'includeMapResults': True, 'isLeanTreatment': False}, 'extensions': {
        'persistedQuery': {'version': 1,
                           'sha256Hash': '81c26682ee29edbbf0cd22db48b9b01b5686c4cb43f2c98758395a0cdac50700'}}}


def data_from_response(r: requests.Response) -> dict[str, dict[str, str]]:
    r_json = r.json()
    try:

        results = r_json["data"]["presentation"]["staysSearch"]["results"]
        if "paginationInfo" in results and "nextPageCursor" in results["paginationInfo"]:
            next_page_cursor = results["paginationInfo"]["nextPageCursor"]
        else:
            next_page_cursor = None

        data = {
            result["listing"]["id"]: {
                "city": result["listing"]["city"],
                "lat": result["listing"]["coordinate"]["latitude"],
                "lon": result["listing"]["coordinate"]["longitude"],
                "type": result["listing"]["roomTypeCategory"],
                "title": result["listing"]["name"],
                "id": result["listing"]["id"],
                "price": result["pricingQuote"]["structuredStayDisplayPrice"]["primaryLine"]#['price']
            }
            for result in results["searchResults"]
        }

    except KeyError as e:
        print("Unexpected JSON format:")
        print(e)
        return {}, None
    except TypeError as e:
        print(e)
        return {}, None

    return data, next_page_cursor


def get_data(location: str, price_min: int) -> list[str]:
    data: dict[str, dict[str, str]] = {}
    # print(location, price_min)
    r = requests.post(url, json=get_request_body(location, price_min), headers=headers)
    first_data, next_page_cursor = data_from_response(r)
    # Merge the dictionaries. We're not concerned with which one wins.
    data = data | first_data

    while next_page_cursor is not None:
        body = get_request_body(location, price_min)
        body["variables"]["staysSearchRequest"]["cursor"] = next_page_cursor

        r = requests.post(url, json=body, headers=headers)
        new_data, next_page_cursor = data_from_response(r)
        data = data | new_data
        # print("sleeping a sec")
        time.sleep(0.5)

    # print(f"Num offers for price: {price_min}: {len(data.keys())}")
    return list(data.keys())

def search_location(name: str, country: str, prices: list[float] = [4000, 3800, 3600, 3400, 3200, 3000, 2800, 2600, 2400, 2200,
                                                                    2000, 1900, 1700, 1500, 1350, 1250, 1100, 1000,
                                                                    800, 700, 600]) -> list[str]:
    # [2000, 1500, 1000, 500]
    location = f"{name}, {country}"
    all_indices = []
    for price_min in prices:
        new_indices = get_data(location, price_min)
        all_indices.extend(new_indices)
    return all_indices


def search_locations(location_file: Union[Path, str]):
    # print(location_file)

    with open(location_file, "r") as fid:
        num_rows = sum(1 for _ in fid.readlines())

    with open(location_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=",", fieldnames=("name", "dest"))
        for idx, row in enumerate(tqdm(reader, total=num_rows)):
            if idx < 83:
                continue
            indices = search_location(row["name"], row["dest"])
            indices = list(set(indices))
            print(f"Num offers for location {row['name']} {row['dest']}: {len(indices)}")
            with open("listings.txt", "a") as fid:
                fid.write("\n".join([str(l) for l in indices]))


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    # search_location("new york city", Path("test"), 50)
    search_locations(args.locations)