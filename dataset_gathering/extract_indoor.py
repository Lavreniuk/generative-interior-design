"""
Create a txt file with indoor pair of listing_id-photo_id
"""
from pathlib import Path
import csv
import base64
from multiprocessing import Pool
import sys
from glob import glob
from typing import List, Optional, Iterator, Dict, Union, Tuple, Any, Iterable
import pickle
import lmdb
import pickle
from tqdm.auto import tqdm
import argtyped
import numpy as np
from helpers import save_txt, get_key

csv.field_size_limit(sys.maxsize)


class Arguments(argtyped.Arguments):
    output: str
    detection: Path = Path("places365")


def load_outdoor(detection: Path) -> List[Tuple[int, int, bool]]:
    is_indoor = []
    for path in Path(detection).glob("*.tsv"):
        with open(path) as fid:
            reader = csv.DictReader(
                fid,
                delimiter="\t",
                fieldnames=["listing_id", "photo_id", "cat", "attr", "outdoor"],
            )
            for row in reader:
                outdoor = eval(row["outdoor"])[1]
                is_indoor.append(
                    (
                        int(row["listing_id"]),
                        int(row["photo_id"]),
                        bool(outdoor),
                    )
                )
    return is_indoor


if __name__ == "__main__":
    args = Arguments()
    print(args.to_string())

    is_indoor = load_outdoor(args.detection)

    keys = [get_key(i[0], i[1]) for i in is_indoor if i[2]]
    save_txt(keys, args.output)
