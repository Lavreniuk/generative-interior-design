"""
Get attributes about images
Inspired by https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py
"""
from pathlib import Path
import os
from typing import List, Iterator, Tuple, Optional, Union, Dict
import hashlib
import json
from multiprocessing import Pool
import urllib.request
import sys
import csv
from tqdm.auto import tqdm
import torch
from torchvision import transforms as trn
from torch import nn
from torch.utils.data._utils.collate import default_collate
from torch.nn import functional as F
import numpy as np
import cv2
from PIL import Image
import tap
from torch.utils.data import Dataset, DataLoader
import wideresnet as wideresnet


csv.field_size_limit(sys.maxsize)

TSV_FIELDNAMES = [
    "listing_id",
    "photo_id",
    "category",
    "attributes",
    "is_indoor",
]


class Arguments(tap.Tap):
    output: Path = Path("places365/detect.tsv")
    images: Path = Path("images")
    batch_size: int = 100
    visualize: bool = False
    num_cat: int = 5
    num_attr: int = 10
    num_splits: int = 1
    start: int = 0
    num_workers: int = 0


# hacky way to deal with the Pytorch 1.0 update
def recursion_change_bn(module: nn.Module) -> nn.Module:
    if isinstance(module, nn.BatchNorm2d):
        module.track_running_stats = 1  # type: ignore
    else:
        for i, (name, module1) in enumerate(module._modules.items()):  # type: ignore
            module1 = recursion_change_bn(module1)
    return module


def download_url(url, cache_dir):
    stem = hashlib.sha1(str(url).encode())
    filename = cache_dir / stem.hexdigest()
    if not filename.is_file():
        urllib.request.urlretrieve(url, filename)
    return filename


def load_labels(
    cache_dir: Union[Path, str]
) -> Tuple[Tuple[str, ...], np.ndarray, List[str], np.ndarray]:
    """
    prepare all the labels
    """

    # indoor and outdoor relevant
    filename_io = download_url(
        "https://raw.githubusercontent.com/csailvision/places365/master/IO_places365.txt",
        cache_dir,
    )
    with open(filename_io) as f:
        lines = f.readlines()
        labels_IO = []
        for line in lines:
            items = line.rstrip().split()
            labels_IO.append(int(items[-1]) - 1)  # 0 is indoor, 1 is outdoor
    labels_IO = np.array(labels_IO)

    # scene category relevant
    filename_category = download_url(
        "https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt",
        cache_dir,
    )
    _classes = list()
    with open(filename_category) as class_file:
        for line in class_file:
            _classes.append(line.strip().split(" ")[0][3:])
    classes = tuple(_classes)

    # scene attribute relevant
    filename_attribute = download_url(
        "https://raw.githubusercontent.com/csailvision/places365/master/labels_sunattribute.txt",
        cache_dir,
    )
    with open(filename_attribute) as f:
        lines = f.readlines()
        labels_attribute = [item.rstrip() for item in lines]

    filename_W = download_url(
        "http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy",
        cache_dir,
    )
    W_attribute = np.load(filename_W)

    return classes, labels_IO, labels_attribute, W_attribute


def get_tf():
    # load the image transformer
    tf = trn.Compose(
        [
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return tf


class NormalizeInverse(trn.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        std_inv = 1 / (std + 1e-7)  # type: ignore
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, array: np.ndarray):
        tensor = torch.tensor(array)
        tensor = super().__call__(tensor.clone())
        array = np.transpose(np.uint8(255 * tensor.numpy()), (1, 2, 0))

        return array


class Hooker:
    def __init__(self, model: nn.Module, features_names=("layer4", "avgpool")):
        self.features: List[np.ndarray] = []

        # this is the last conv layer of the resnet
        for name in features_names:
            model._modules.get(name).register_forward_hook(self)  # type: ignore

    def __call__(self, module: nn.Module, input, output):
        self.features.append(output.data.cpu().numpy())

    def reset(self):
        self.features = []


# load the model
def load_model(cache_dir: Union[Path, str]) -> nn.Module:
    # this model has a last conv feature map as 14x14

    model_file = download_url(
        "http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar",
        cache_dir,
    )

    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)

    # hacky way to deal with the upgraded batchnorm2D and avgpool layers...
    for i, (name, module) in enumerate(model._modules.items()):  # type: ignore
        module = recursion_change_bn(model)  # type: ignore
    model.avgpool = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)  # type: ignore

    model.eval()
    return model


def search_locations(image_folder: Path) -> List[Path]:
    return [f for f in image_folder.iterdir() if f.is_dir()]


def load_photo_paths(locations: List[Path]) -> Iterator[Path]:
    for location in tqdm(locations):
        for photo in location.glob("*.jpg"):
            yield photo


def load_photos(images: Path, cache_dir: Union[Path, str]) -> List[Union[str, Path]]:
    photo_cache = Path(cache_dir) / "photos.txt"

    if photo_cache.is_file():
        with open(photo_cache, "r") as fid:
            photos: List[Union[str, Path]] = [l.strip() for l in fid.readlines()]
    else:
        print("Preloading every images")
        photos = list(images.rglob("*.jpg"))
        with open(photo_cache, "w") as fid:
            fid.writelines(f"{l}\n" for l in photos)

    return photos


class ImageDataset(Dataset):
    def __init__(self, photos: List[Union[Path, str]]):
        self.photos = photos
        self.tf = get_tf()  # image transformer

    def __len__(self):
        return len(self.photos)

    def __getitem__(
        self, index: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        path = Path(self.photos[index])

        try:
            image = Image.open(path)
            image = image.convert("RGB")
        except:
            return None
        tensor = self.tf(image)

        listing_id, photo_id = map(int, path.stem.split("-"))
        return torch.tensor(listing_id), torch.tensor(photo_id), tensor


def collate_fn(batch: Tuple):
    batch = tuple([b for b in batch if b is not None])

    if not batch:
        return None

    return default_collate(batch)


def class_activation_map(
    feature_conv: np.ndarray, weight_softmax: np.ndarray, class_idx: List[int]
):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    for _ in class_idx:
        cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))  # type: ignore
    return output_cam


def get_key(listing_id, photo_id) -> str:
    return f"{listing_id}_{photo_id}"


def is_indoor(idx, labels_io):
    # vote for the indoor or outdoor
    io_image = np.mean(labels_io[idx[:10]])
    ans = bool(io_image < 0.5)
    return io_image, ans


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # type: ignore


@torch.no_grad()
def run_model(
    batch: List[torch.Tensor],
    model,
    hook,
    classes: Tuple[str, ...],
    labels_IO: np.ndarray,
    labels_attribute: List[str],
    W_attribute: np.ndarray,
    num_cat: int,
    num_attr: int,
    weight_softmax: Optional[np.ndarray] = None,
) -> List[Dict]:
    listing_ids, photo_ids, input_img = batch

    # forward pass
    logit = model.forward(input_img.cuda())
    h_x = F.softmax(logit, 1)

    detections = []

    for i, p in enumerate(h_x):  # type: ignore
        listing_id = int(listing_ids[i])
        photo_id = int(photo_ids[i])
        key = get_key(listing_id, photo_id)

        probs, idx = p.sort(0, True)  # type: ignore
        probs = probs.detach().cpu().numpy()
        idx = idx.detach().cpu().numpy()

        # scene category
        category = [(probs[j], classes[idx[j]]) for j in range(0, num_cat)]

        # output the scene attributes
        ft = [np.squeeze(f[i]) for f in hook.features]
        responses_attribute = softmax(W_attribute.dot(ft[1]))
        idx_a = np.argsort(responses_attribute)
        attributes = [
            (responses_attribute[idx_a[j]], labels_attribute[idx_a[j]])
            for j in range(-1, -num_attr, -1)
        ]

        detections.append(
            {
                "listing_id": listing_id,
                "photo_id": photo_id,
                "category": category,
                "attributes": attributes,
                "is_indoor": is_indoor(idx, labels_IO),
            }
        )

        # generate class activation mapping
        if weight_softmax is not None:
            ca_map = class_activation_map(ft[0], weight_softmax, [idx[0]])[0]

            # render the CAM and output
            img = NormalizeInverse()(input_img[i])
            height, width, _ = img.shape  # type: ignore
            heatmap = cv2.applyColorMap(  # type: ignore
                cv2.resize(ca_map, (width, height)), cv2.COLORMAP_JET  # type: ignore
            )
            result = heatmap * 0.4 + img * 0.5  # type: ignore
            cv2.imwrite(f"examples/{key}-heatmap.jpg", result)  # type: ignore
            cv2.imwrite(f"examples/{key}-image.jpg", img[:, :, ::-1])  # type: ignore

    hook.reset()

    return detections


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_json(data, filename: Union[str, Path]):
    with open(filename, "w") as fid:
        json.dump(data, fid, indent=2, cls=NumpyEncoder)


def detection(args: Arguments, proc_id: int, cache_dir: Union[Path, str]):
    # load the labels
    classes, labels_IO, labels_attribute, W_attribute = load_labels(cache_dir)

    model = load_model(cache_dir)
    hook = Hooker(model)
    # load the transformer
    # get the softmax weight
    params = list(model.parameters())

    if args.visualize:
        weight_softmax = params[-2].data.numpy()
        weight_softmax[weight_softmax < 0] = 0
    else:
        weight_softmax = None

    photos = load_photos(args.images, cache_dir)
    print("The dataset contains a total of", len(photos))
    photos = photos[proc_id :: args.num_splits]
    print(
        "The split", proc_id, "over", args.num_splits, "contains", len(photos), "photos"
    )

    dataset = ImageDataset(photos)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_fn,  # type: ignore
    )

    model = model.cuda()

    args.output.parent.mkdir(exist_ok=True, parents=True)
    filename = args.output.parent / f"{args.output.stem}.{proc_id}.tsv"
    print(f"Start split {proc_id} on {len(dataset)} photos")
    with open(filename, "wt") as tsvfile:
        writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=TSV_FIELDNAMES)
        for batch in tqdm(dataloader):
            if batch is None:
                continue
            detections = run_model(
                batch,
                model,
                hook,
                classes,
                labels_IO,
                labels_attribute,
                W_attribute,
                num_cat=args.num_cat,
                num_attr=args.num_attr,
                weight_softmax=weight_softmax,
            )
            for d in detections:
                writer.writerow(d)


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    cache_dir = Path.home() / ".cache" / args.output.parent.name
    cache_dir.mkdir(exist_ok=True, parents=True)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    start = max(local_rank, 0) + args.start
    detection(args, start, cache_dir)
