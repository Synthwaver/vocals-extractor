import argparse
import csv
import os
import pathlib
from typing import Collection

import pandas


def make_csv(
        csv_filepath: str,
        target_dir: str,
        columns: Collection[str] = None,
        separator: str = ",",
        absolute: bool = False
) -> None:
    if columns is None:
        columns = ["mixture", "vocals", "accompaniment"]

    target_dir = pathlib.Path(target_dir)
    if not target_dir.is_dir():
        raise NotADirectoryError(target_dir)

    csv_filepath = pathlib.Path(csv_filepath)
    csv_dir = csv_filepath.absolute().parent
    df = pandas.DataFrame(columns=columns)

    sub_dirs = next(os.walk(target_dir))[1]
    for sub_dir in sub_dirs:
        result_dir = pathlib.Path(target_dir, sub_dir).absolute()
        if not absolute:
            try:
                relative = result_dir.relative_to(csv_dir)
            except ValueError:
                result_dir = result_dir.absolute()
            else:
                result_dir = relative

        files = next(os.walk(pathlib.Path(target_dir, sub_dir)))[2]
        row = dict()
        for file in files:
            name = os.path.splitext(file)[0]
            if name in columns:
                row[name] = pathlib.Path(result_dir, file).as_posix()
        if len(row) > 0:
            df = df.append(row, ignore_index=True)

    df.to_csv(csv_filepath, sep=separator, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file')
    parser.add_argument('dir')
    parser.add_argument('-c', '--columns', nargs='+')
    parser.add_argument('-s', '--separator', default=',')
    parser.add_argument('-a', '--absolute', action='store_const', const=True)
    namespace = parser.parse_args()

    make_csv(namespace.file, namespace.dir,
             columns=namespace.columns,
             separator=namespace.separator,
             absolute=namespace.absolute)
