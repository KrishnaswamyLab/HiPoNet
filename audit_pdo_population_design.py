#!/usr/bin/env python
"""Report designed PDO/CAF populations absent from the released cell table."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


GROUP_KEYS = (
    "Patient",
    "Culture",
    "Treatment",
    "Concentration",
    "Replicate",
    "Cell_type",
)
DESIGN_ARMS = (
    ("PDO", "PDOs"),
    ("F", "Fibs"),
    ("PDOF", "PDOs"),
    ("PDOF", "Fibs"),
)


def groups_from_cache(path: Path) -> set[tuple[str, ...]]:
    cache = np.load(path, allow_pickle=True)
    keys = tuple(map(str, cache["group_keys"]))
    if keys != GROUP_KEYS:
        raise ValueError(f"Expected group keys {GROUP_KEYS}, found {keys}")
    return {tuple(str(name).split("__")) for name in cache["group_names"]}


def groups_from_zip(path: Path, member: str) -> set[tuple[str, ...]]:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            frame = pd.read_pickle(handle)
    missing_columns = [key for key in GROUP_KEYS if key not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing metadata columns: {missing_columns}")
    return {
        tuple(map(str, row))
        for row in frame.loc[:, GROUP_KEYS].drop_duplicates().itertuples(
            index=False, name=None
        )
    }


def expected_groups(observed: set[tuple[str, ...]]) -> set[tuple[str, ...]]:
    positions = {key: index for index, key in enumerate(GROUP_KEYS)}
    patients = {row[positions["Patient"]] for row in observed}
    conditions = {
        (
            row[positions["Treatment"]],
            row[positions["Concentration"]],
            row[positions["Replicate"]],
        )
        for row in observed
        if row[positions["Culture"]] == "PDOF"
        and row[positions["Cell_type"]] == "PDOs"
    }
    return {
        (patient, culture, treatment, concentration, replicate, cell_type)
        for patient in patients
        for culture, cell_type in DESIGN_ARMS
        for treatment, concentration, replicate in conditions
    }


def sort_key(group: tuple[str, ...]) -> tuple:
    patient = int(group[0]) if group[0].isdigit() else group[0]
    return patient, group[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--population-cache", type=Path)
    source.add_argument("--zip-path", type=Path)
    parser.add_argument("--member", default="Metadata_final_paper")
    parser.add_argument("--show-extra", action="store_true")
    args = parser.parse_args()

    observed = (
        groups_from_cache(args.population_cache)
        if args.population_cache is not None
        else groups_from_zip(args.zip_path, args.member)
    )
    expected = expected_groups(observed)
    missing = sorted(expected - observed, key=sort_key)
    extra = sorted(observed - expected, key=sort_key)

    print(f"Designed groups: {len(expected)}")
    print(f"Observed nonempty groups: {len(observed)}")
    print(f"Missing designed groups: {len(missing)}")
    for group in missing:
        print("__".join(group))
    if args.show_extra:
        print(f"Unexpected additional groups: {len(extra)}")
        for group in extra:
            print("__".join(group))


if __name__ == "__main__":
    main()
