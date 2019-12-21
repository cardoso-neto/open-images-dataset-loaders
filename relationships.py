
from itertools import chain
from operator import attrgetter, contains, itemgetter
from pathlib import Path
from typing import Callable

import torch
from bidict import bidict
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import csv_to_dict, multicolumn_csv_to_dict, read_csv


VRD_INDICES = {
    'ImageID': 0,
    'LabelName1': 1,
    'LabelName2': 2,
    'XMin1': 3,
    'XMax1': 4,
    'YMin1': 5,
    'YMax1': 6,
    'XMin2': 7,
    'XMax2': 8,
    'YMin2': 9,
    'YMax2': 10,
    'RelationshipLabel': 11,
}

TRIPLETS_INDICES = {
    "subject": 0,
    "object": 1,
    "relationship": 2,
}


class ImageAndRelationships(Dataset):
    def __init__(
        self,
        root_folder: Path,
        split: str = "validation",
        transform: Callable = None,
    ):
        """
        Visual Relationship Detection dataset.

        [extended_summary]

        :param root_folder:
        :param split:
        :param transform:
        """
        super().__init__()
        self.transform = transform
        metadata_folder = root_folder / "annotations" / "metadata"

        images_folder = root_folder / "images"
        if split == "train":
            all_folders = images_folder.glob(f"{split}_0" + r"[0-9]")
            all_images = chain(
                *[folder.glob(r"*.jpg") for folder in all_folders]
            )
        else:
            img_folder = images_folder / split
            all_images = img_folder.glob(r"*.jpg")

        vrd_csv_filepath = root_folder / "annotations" / "relationships"
        if split == "train":
            vrd_csv_filepath /= f"challenge-2018-{split}-vrd.csv"
        else:
            vrd_csv_filepath /= f"{split}-annotations-vrd.csv"

        self.images_to_relationships = multicolumn_csv_to_dict(
            vrd_csv_filepath, one_to_n_mapping=True
        )
        current_split = set(self.images_to_relationships.keys())
        self.images = [
            image_path for image_path in all_images
            if image_path.stem in current_split
        ]

        self.label_name_to_class_description = csv_to_dict(
            metadata_folder / "class-descriptions-boxable.csv",
            discard_header=False,
        )
        self.label_name_to_class_description_extension = csv_to_dict(
            metadata_folder / "challenge-2018-attributes-description.csv",
            discard_header=False,
        )
        self.label_name_to_class_description.update(
            self.label_name_to_class_description_extension
        )

        self.label_name_to_id = bidict(
            zip(
                self.label_name_to_class_description.keys(),
                range(len(self.label_name_to_class_description.keys())),
            )
        )
        triplets = read_csv(
            metadata_folder / "challenge-2018-relationship-triplets.csv"
        )
        relationship_names = sorted(
            set(
                map(itemgetter(TRIPLETS_INDICES["relationship"]), triplets)
            )
        )
        self.relationship_names_to_id = bidict(
            zip(relationship_names, range(len(relationship_names)))
        )

    def get_stats(self):
        # TODO: this is used to work before the switch to defaultdict(list)
        stats = {
            "relationship_types": len(self.relationship_names_to_id),
            "unique_subjects": len(
                set(
                    map(
                        itemgetter(VRD_INDICES["LabelName1"] - 1),
                        self.images_to_relationships.values(),
                    )
                )
            ),
            "unique_objects": len(
                set(
                    map(
                        itemgetter(VRD_INDICES["LabelName2"] - 1),
                        self.images_to_relationships.values(),
                    )
                )
            ),
            "images": len(self.images),
            "relationships": len(self.images_to_relationships)
        }
        return stats

    def get_readable_labels(self):
        # TODO: this is used to work before the switch to defaultdict(list)
        unique_subjects = set(
            map(
                itemgetter(VRD_INDICES["LabelName1"] - 1),
                self.images_to_relationships.values(),
            )
        )
        unique_subjects = {
            self.label_name_to_class_description[label_name]
            for label_name in unique_subjects
        }
        unique_objects = set(
            map(
                itemgetter(VRD_INDICES["LabelName2"] - 1),
                self.images_to_relationships.values(),
            )
        )
        unique_objects = {
            self.label_name_to_class_description[label_name]
            for label_name in unique_objects
        }
        data = {
            "unique_subjects": unique_subjects,
            "unique_objects": unique_objects,
        }
        return data

    def __len__(self) -> int:
        return len(self.images)

    def prep_labels(self, labels):
        obj_1, obj_2, *bboxes, relationship_name = labels
        obj_id_1 = torch.tensor(self.label_name_to_id[obj_1])
        obj_id_2 = torch.tensor(self.label_name_to_id[obj_2])
        bboxes = torch.tensor(list(map(float, bboxes)))
        bbox_1, bbox_2 = bboxes[:4], bboxes[4:]
        relationship_id = self.relationship_names_to_id[relationship_name]
        relationship_id = torch.tensor(relationship_id)
        return (obj_id_1, obj_id_2, bbox_1, bbox_2, relationship_id)

    def prep_images(self, image, labels):
        # TODO: fix crops on wrongs dims
        sub = []
        obj = []
        comb = []

        for label in labels:
            bbox_sub = label[2]
            sub.append(
                image[
                    bbox_sub[0]:bbox_sub[1],
                    bbox_sub[2]:bbox_sub[3],
                ]
            )

            bbox_obj = label[3]
            obj.append(
                image[
                    bbox_obj[0]:bbox_obj[1],
                    bbox_obj[2]:bbox_obj[3],
                ]
            )

            bbox_comb = [
                min(bbox_sub[0], bbox_obj[0]),
                max(bbox_sub[1], bbox_obj[1]),
                min(bbox_sub[3], bbox_obj[3]),
                max(bbox_sub[4], bbox_obj[4]),
            ]
            comb.append(
                image[
                    bbox_comb[0]:bbox_comb[1],
                    bbox_comb[2]:bbox_comb[3],
                ]
            )

        return sub, obj, comb

    def __getitem__(self, index: int):
        instance, *labels = self.instances[index]
        image_path = self.image_paths[instance]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image.convert("RGB")
        labels = self.prep_labels(labels)

        crops = prep_images(image, labels)

        if self.transform:
            crops = self.transform(crops)

        return crops, labels[:, -1]


class Relationships(Dataset):
    def __init__(
        self,
        root_folder: Path,
        split: str = "validation",
        transform: Callable = None,
    ):
        """
        Visual Relationship Detection dataset.

        [extended_summary]

        :param root_folder:
        :param split:
        :param transform:
        """
        super().__init__()
        self.transform = transform
        metadata_folder = root_folder / "annotations" / "metadata"

        images_folder = root_folder / "images"
        if split == "train":
            all_folders = images_folder.glob(f"{split}_*")
            print(*all_folders)
            exit()
            all_images = chain(
                *[folder.glob(r"*.jpg") for folder in all_folders]
            )
        else:
            img_folder = images_folder / split
            all_images = img_folder.glob(r"*.jpg")

        vrd_csv_filepath = root_folder / "annotations" / "relationships"
        if split == "train":
            vrd_csv_filepath /= f"challenge-2018-{split}-vrd.csv"
        else:
            vrd_csv_filepath /= f"{split}-annotations-vrd.csv"

        self.instances = read_csv(vrd_csv_filepath)

        current_split = set(
            map(
                itemgetter(VRD_INDICES["ImageID"]),
                self.instances,
            )
        )

        self.image_paths = {
            image_path.stem: image_path for image_path in all_images
            if image_path.stem in current_split
        }

        self.label_name_to_class_description = csv_to_dict(
            metadata_folder / "class-descriptions-boxable.csv",
            discard_header=False,
        )
        self.label_name_to_class_description_extension = csv_to_dict(
            metadata_folder / "challenge-2018-attributes-description.csv",
            discard_header=False,
        )
        self.label_name_to_class_description.update(
            self.label_name_to_class_description_extension
        )

        self.label_name_to_id = bidict(
            zip(
                self.label_name_to_class_description.keys(),
                range(len(self.label_name_to_class_description.keys())),
            )
        )
        triplets = read_csv(
            metadata_folder / "challenge-2018-relationship-triplets.csv"
        )
        relationship_names = sorted(
            set(
                map(itemgetter(TRIPLETS_INDICES["relationship"]), triplets)
            )
        )
        self.relationship_names_to_id = bidict(
            zip(relationship_names, range(len(relationship_names)))
        )

    def __len__(self) -> int:
        return len(self.instances)

    def prep_labels(self, labels):
        obj_1, obj_2, *bboxes, relationship_name = labels
        obj_id_1 = torch.tensor(self.label_name_to_id[obj_1])
        obj_id_2 = torch.tensor(self.label_name_to_id[obj_2])
        bboxes = torch.tensor(list(map(float, bboxes)))
        bbox_1, bbox_2 = bboxes[:4], bboxes[4:]
        relationship_id = self.relationship_names_to_id[relationship_name]
        relationship_id = torch.tensor(relationship_id)
        return (obj_id_1, obj_id_2, bbox_1, bbox_2, relationship_id)

    def prep_images(self, image, labels):
        # TODO: crop images using PIL to avoid conversion to np.ndarray
        h, w = image.shape[:2]

        bbox_sub = (labels[2] * torch.tensor([w, w, h, h])).int()
        sub = image[
            bbox_sub[2]:bbox_sub[3],
            bbox_sub[0]:bbox_sub[1],
        ]

        bbox_obj = (labels[3] * torch.tensor([w, w, h, h])).int()
        obj = image[
            bbox_obj[2]:bbox_obj[3],
            bbox_obj[0]:bbox_obj[1],
        ]

        bbox_comb = [
            min(bbox_sub[0], bbox_obj[0]),
            max(bbox_sub[1], bbox_obj[1]),
            min(bbox_sub[2], bbox_obj[2]),
            max(bbox_sub[3], bbox_obj[3]),
        ]
        comb = image[
            bbox_comb[2]:bbox_comb[3],
            bbox_comb[0]:bbox_comb[1],
        ]

        sub = Image.fromarray(sub)
        obj = Image.fromarray(obj)
        comb = Image.fromarray(comb)

        crops = (sub, obj, comb)
        return crops

    def __getitem__(self, index: int):
        instance, *labels = self.instances[index]
        image_path = self.image_paths[instance]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        labels = self.prep_labels(labels)

        image = np.asarray(image)
        crops = self.prep_images(image, labels)
        if self.transform:
            crops = tuple(self.transform(x) for x in crops)

        return crops, (labels[0], labels[1], labels[-1])


if __name__ == "__main__":
    a = ImageAndRelationships(Path("../../open-images"), split="test")
    print([a[i] for i in range(10)])
    a = Relationships(Path("../../open-images"), split="test")
    print([a[i] for i in range(10)])
