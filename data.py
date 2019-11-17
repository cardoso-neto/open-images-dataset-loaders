
from operator import attrgetter, contains, itemgetter
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import csv_to_dict, multicolumn_csv_to_dict, read_csv


BBOX_CSV_HEADER = {
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
        [summary]

        [extended_summary]

        :param root_folder: [description]
        :param split: [description], defaults to "validation"
        :param transform: [description], defaults to None
        """
        super().__init__()
        self.transform = transform
        images_folder = root_folder / "images" / split

        self.box_labels = multicolumn_csv_to_dict(
            root_folder.joinpath(
                "annotations", "boxes", f"{split}-annotations-bbox.csv"
            ),
            value_cols=itemgetter(
                "LabelName",
                "Confidence",
                "XMin",
                "XMax",
                "YMin",
                "YMax",
            )(BBOX_CSV_HEADER),
        )
        current_split = set(self.box_labels.keys())
        self.images = [
            image_path for image_path in images_folder.glob(r"*.jpg")
            if image_path.stem in current_split
        ]
        self.class_names = csv_to_dict(
            root_folder.joinpath(
                "annotations", "metadata", "class-descriptions-boxable.csv"
            )
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        image = Image.open(image_path)

        labels = self.box_labels[image_path.stem]

        if self.transform:
            image = self.transform(image)

        return image, labels


if __name__ == "__main__":
    a = OpenImagesObjects(Path("../../open-images"))
    print(a[10])
