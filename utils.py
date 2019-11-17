
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


def read_file(file_path: Path) -> str:
    with open(file_path) as text_file:
        return text_file.read()


def read_multiline(file_path: Path) -> List[str]:
    """Read, split by LF, and pop final empty line if present."""
    text = read_file(file_path)
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def read_csv(file_path: Path) -> List[List[str]]:
    lines = read_multiline(file_path)
    table = list(map(methodcaller("split", ","), lines))
    return table


def csv_to_dict(
    file_path: Path, key_col: int = 0, value_col: int = 1
) -> Dict[str, str]:
    table = read_csv(file_path)
    return dict(map(itemgetter(key_col, value_col), table))


def multicolumn_csv_to_dict(
    file_path: Path,
    key_cols: Sequence = (0,),
    value_cols: Optional[Sequence] = None,
) -> Dict[str, Tuple[str]]:
    table = read_csv(file_path)
    if not value_cols:
        value_cols = tuple(i for i in range(1, len(table[0]) - 1))
    key_columns = map(itemgetter(*key_cols), table)
    value_columns = map(itemgetter(*value_cols), table)
    return dict(zip(key_columns, value_columns))
