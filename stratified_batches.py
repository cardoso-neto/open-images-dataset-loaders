
from pathlib import Path
from typing import Callable

from relationships import Relationships


class StratifiedRelationships(Relationships):
    def __init__(
        self,
        root_folder: Path,
        split: str = "validation",
        transform: Callable = None,
    ):
        super().__init__(root_folder, split, transform)
        self.images_to_relationships
