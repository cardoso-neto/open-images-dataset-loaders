
from collections import defaultdict
from operator import itemgetter, methodcaller
from pathlib import Path
from typing import (
    DefaultDict,
    Dict,
    List,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)


K = TypeVar("K")
V = TypeVar("V")

def default_dict(pairs: Iterable[Tuple[K, V]]) -> DefaultDict[K, V]:
    mapping = defaultdict(list)
    for key, val in pairs:
        mapping[key].append(val)
    return mapping


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


def read_csv(file_path: Path, discard_header: bool = True) -> List[List[str]]:
    lines = read_multiline(file_path)
    print(file_path, len(lines))
    table = map(methodcaller("split", ","), lines)
    if discard_header:
        next(table)
    return list(table)


def csv_to_dict(
    file_path: Path,
    key_col: int = 0,
    value_col: int = 1,
    discard_header: bool = True,
    one_to_n_mapping: bool = False,
) -> Union[Dict[str, str], DefaultDict[str, str]]:
    table = read_csv(file_path, discard_header)
    pairs = map(itemgetter(key_col, value_col), table)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)


def multicolumn_csv_to_dict(
    file_path: Path,
    key_cols: Sequence = (0,),
    value_cols: Optional[Sequence] = None,
    discard_header: bool = True,
    one_to_n_mapping: bool = False,
) -> Union[Dict[str, Tuple[str]], DefaultDict[str, Tuple[str]]]:
    # TODO fix return type: keys can also be tuples...
    table = read_csv(file_path, discard_header)
    if not value_cols:
        value_cols = tuple(i for i in range(1, len(table[0])))
    key_columns = map(itemgetter(*key_cols), table)
    value_columns = map(itemgetter(*value_cols), table)
    pairs = zip(key_columns, value_columns)
    if one_to_n_mapping:
        return default_dict(pairs)
    return dict(pairs)
