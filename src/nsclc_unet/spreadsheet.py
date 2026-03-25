from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as ET

import pandas as pd


XML_NS = {
    "a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _column_index(cell_ref: str) -> int:
    letters = "".join(character for character in cell_ref if character.isalpha())
    index = 0
    for character in letters:
        index = index * 26 + (ord(character.upper()) - 64)
    return index - 1


def _make_unique_headers(headers: list[str]) -> list[str]:
    counts: defaultdict[str, int] = defaultdict(int)
    unique_headers: list[str] = []
    for index, header in enumerate(headers):
        candidate = str(header).strip() if header is not None else ""
        if not candidate:
            candidate = f"column_{index}"
        counts[candidate] += 1
        if counts[candidate] > 1:
            unique_headers.append(f"{candidate}__{counts[candidate]}")
        else:
            unique_headers.append(candidate)
    return unique_headers


def _shared_strings(archive: ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []

    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    strings: list[str] = []
    for node in root.findall("a:si", XML_NS):
        strings.append("".join(token.text or "" for token in node.iterfind(".//a:t", XML_NS)))
    return strings


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    value_node = cell.find("a:v", XML_NS)
    if value_node is None:
        inline = cell.find("a:is", XML_NS)
        if inline is None:
            return ""
        return "".join(token.text or "" for token in inline.iterfind(".//a:t", XML_NS))

    value = value_node.text or ""
    if cell_type == "s" and value.isdigit():
        index = int(value)
        if 0 <= index < len(shared_strings):
            return shared_strings[index]
    return value


def read_xlsx_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    with ZipFile(path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relationship_map = {
            relation.attrib["Id"]: relation.attrib["Target"] for relation in relationships
        }
        shared_strings = _shared_strings(archive)

        sheet = next(
            candidate
            for candidate in workbook.findall("a:sheets/a:sheet", XML_NS)
            if candidate.attrib["name"] == sheet_name
        )
        relationship_id = sheet.attrib[
            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
        ]
        target = relationship_map[relationship_id]
        root = ET.fromstring(archive.read(f"xl/{target}"))

        sparse_rows: list[dict[int, str]] = []
        max_columns = 0
        for row in root.findall(".//a:sheetData/a:row", XML_NS):
            row_map: dict[int, str] = {}
            for cell in row.findall("a:c", XML_NS):
                index = _column_index(cell.attrib["r"])
                row_map[index] = _cell_value(cell, shared_strings)
                max_columns = max(max_columns, index + 1)
            sparse_rows.append(row_map)

        if not sparse_rows:
            raise ValueError(f"Sheet '{sheet_name}' is empty in {path}")

        dense_rows = [[row.get(column_index, "") for column_index in range(max_columns)] for row in sparse_rows]
        headers = _make_unique_headers(dense_rows[0])
        return pd.DataFrame(dense_rows[1:], columns=headers)

