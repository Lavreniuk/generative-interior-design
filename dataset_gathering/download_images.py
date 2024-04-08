"""
Download list of rooms
"""
import csv
from multiprocessing import Pool
from pathlib import Path
from tqdm.auto import tqdm
import tap
from helpers import download_file, DownloadError


class Arguments(tap.Tap):
    csv_file: Path
    output: Path
    correspondance: Path
    num_procs: int = 0
    num_parts: int = 5
    num_splits: int = 1
    start: int = 0


def download_photos(csv_file):
    worker_id = int(Path(csv_file).stem.split("-")[0])
    print(csv_file, worker_id)

    # if worker_id in (0, 1, 10, 11, 12, 13, 14, 19, 21, 22, 23, 25, 27, 4, 5, 8):
    #     return

    with open(csv_file, newline="") as fid:
        fieldnames = ["url", "path"]
        reader = csv.DictReader(fid, fieldnames=fieldnames, delimiter="\t")
        num_rows = 0
        for _ in reader:
            num_rows += 1

    with open(csv_file, newline="") as fid:
        fieldnames = ["url", "path"]
        reader = csv.DictReader(fid, fieldnames=fieldnames, delimiter="\t")

        for row in tqdm(
            reader,
            desc="#{0}: ".format(worker_id),
            position=worker_id,
            total=num_rows,
            leave=True,
        ):
            Path(row["path"]).parent.mkdir(exist_ok=True)
            try:
                download_file(row["url"], row["path"])
            except DownloadError:
                with open("errors.txt", "a") as fid:
                    fid.write(f"\n{row['url']}")


def run_downloader(args: Arguments):
    """
    Inputs:
        process: (int) number of process to run
        images_url:(list) list of images url
    """
    print(f"Running {args.num_procs} workers")

    parts = [
        args.correspondance / f"{str(i)}-{str(args.num_splits)}.csv"
        for i in range(args.num_splits)
    ]
    parts = parts[args.start::args.num_parts]

    if args.num_procs > 0:
        with Pool(args.num_procs) as pool:
            pool.map(download_photos, parts)
    else:
        for part in parts:
            download_photos(part)


def make_correspondance(args: Arguments):
    """
    CSV file: url\t path/to/location
    """
    print(f"Running {args.num_splits} splits")
    args.correspondance.mkdir(parents=True)

    # Load rows
    with open(args.csv_file, newline="") as fid:
        fieldnames = ["listing_id", "photo_id", "url", "caption"]
        rows = list(csv.DictReader(fid, fieldnames=fieldnames, delimiter="\t"))
    num_photos = len(rows)
    print(f"Found {num_photos} photos")

    # Dispatch them
    for i in range(args.num_splits):
        fieldnames = ["url", "path"]
        folder = args.output / str(i)
        folder.mkdir(exist_ok=True, parents=True)
        with open(
            args.correspondance / f"{str(i)}-{str(args.num_splits)}.csv",
            "w",
            newline="",
        ) as fid:
            writer = csv.DictWriter(fid, fieldnames=fieldnames, delimiter="\t")

            for row in rows[i :: args.num_splits]:
                path = folder / f"{row['listing_id']}-{row['photo_id']}.jpg"
                writer.writerow({"url": row["url"], "path": str(path)})


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    if not args.correspondance.is_dir():
        print("Making correspondance")
        make_correspondance(args)

    download = args.output
    download.mkdir(exist_ok=True)

    run_downloader(args)
