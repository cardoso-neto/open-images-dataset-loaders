
from itertools import chain
from pathlib import Path
from typing import Callable

import torch
from bidict import bidict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import csv_to_dict, multicolumn_csv_to_dict


BBOX_INDICES = {
    'ImageID': 0,
    'Source': 1,
    'LabelName': 2,
    'Confidence': 3,
    'XMin': 4,
    'XMax': 5,
    'YMin': 6,
    'YMax': 7,
    'IsOccluded': 8,
    'IsTruncated': 9,
    'IsGroupOf': 10,
    'IsDepiction': 11,
    'IsInside': 12,
}


class OpenImagesObjects(Dataset):

    def __init__(
        self,
        root_folder: Path,
        split: str = "validation",
        transform: Callable = None,
    ):
        """
        Object Detection dataset.

        [extended_summary]

        :param root_folder:
        :param split:
        :param transform:
        """
        super().__init__()
        self.transform = transform
        images_folder = root_folder / "images"
        if split == "train":
            all_folders = images_folder.glob(f"{split}_0" + r"[0-9]")
            all_images = chain(
                *[folder.glob(r"*.jpg") for folder in all_folders]
            )
        else:
            images_folder /= split
            all_images = images_folder.glob(r"*.jpg")

        bbox_csv_filepath = root_folder.joinpath(
            "annotations", "boxes", f"{split}-annotations-bbox.csv"
        )
        indices = tuple(
            BBOX_INDICES[key] for key in (
                "LabelName",
                "XMin",
                "XMax",
                "YMin",
                "YMax",
            )
        )
        self.box_labels = multicolumn_csv_to_dict(
            bbox_csv_filepath, value_cols=indices, one_to_n_mapping=True
        )
        images_with_labels = set(self.box_labels.keys())
        self.images = [
            image_path for image_path in all_images
            if image_path.stem in images_with_labels
        ]

        self.label_name_to_class_description = csv_to_dict(
            root_folder.joinpath(
                "annotations", "metadata", "class-descriptions-boxable.csv"
            ),
            discard_header=False,
        )
        self.label_name_to_id = bidict(
            zip(
                self.label_name_to_class_description.keys(),
                range(len(self.label_name_to_class_description.keys())),
            )
        )

    def __len__(self) -> int:
        return len(self.images)

    def prep_labels(self, labels):
        obj_label, *bbox = labels
        obj_id = torch.tensor(self.label_name_to_id[obj_label])
        bbox = torch.tensor(list(map(float, bbox)))
        return (obj_id, bbox)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        labels = self.box_labels[image_path.stem]
        labels = list(map(self.prep_labels, labels))
        return image, labels


if __name__ == "__main__":
    a = OpenImagesObjects(Path("../../open-images"), split="test")
    print([a[i] for i in range(10)])
