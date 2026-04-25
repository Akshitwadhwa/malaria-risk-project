#!/usr/bin/env python3
"""Extract malaria tables from the provided PDF exports into CSV files.

The source files are Excel sheets printed to PDF, so this script uses
`pdftotext -layout` and then applies table-specific cleanup rules.
"""

from __future__ import annotations

import csv
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

DISTRICT_PDF = RAW_DIR / "62270322291720010405.pdf"
STATE_SITUATION_PDF = RAW_DIR / "76430196371770710822.pdf"
EPIDEMIOLOGICAL_PDF = RAW_DIR / "1357043511770710414.pdf"

YEARS = [2021, 2022, 2023, 2024, 2025]

STATE_ALIASES = {
    "Andaman And Nicobar Islands": "Andaman And Nicobar Islands",
    "Andaman And Nicobar": "Andaman And Nicobar Islands",
    "Andhra Pradesh": "Andhra Pradesh",
    "Arunachal Pradesh": "Arunachal Pradesh",
    "Assam": "Assam",
    "Bihar": "Bihar",
    "Chandigarh": "Chandigarh",
    "Chhattisgarh": "Chhattisgarh",
    "D & N Haveli": "The Dadra And Nagar Haveli And Daman And Diu",
    "Daman & Diu": "The Dadra And Nagar Haveli And Daman And Diu",
    "Delhi": "Delhi",
    "Goa": "Goa",
    "Gujarat": "Gujarat",
    "Haryana": "Haryana",
    "Himachal Pradesh": "Himachal Pradesh",
    "Jammu And Kashmir": "Jammu And Kashmir",
    "Jharkhand": "Jharkhand",
    "Karnataka": "Karnataka",
    "Kerala": "Kerala",
    "Ladakh": "Ladakh",
    "Lakshadweep": "Lakshadweep",
    "Madhya Pradesh": "Madhya Pradesh",
    "Maharashtra": "Maharashtra",
    "Manipur": "Manipur",
    "Meghalaya": "Meghalaya",
    "Mizoram": "Mizoram",
    "Nagaland": "Nagaland",
    "Odisha": "Odisha",
    "Puducherry": "Puducherry",
    "Punjab": "Punjab",
    "Rajasthan": "Rajasthan",
    "Sikkim": "Sikkim",
    "Tamil Nadu": "Tamil Nadu",
    "Telangana": "Telangana",
    "The Dadra And Nagar Haveli And Daman And Diu": "The Dadra And Nagar Haveli And Daman And Diu",
    "The Dadra And Nagar": "The Dadra And Nagar Haveli And Daman And Diu",
    "Tripura": "Tripura",
    "Uttar Pradesh": "Uttar Pradesh",
    "Uttarakhand": "Uttarakhand",
    "West Bengal": "West Bengal",
}

ALIASES_BY_LENGTH = sorted(STATE_ALIASES, key=len, reverse=True)

DISTRICT_STATE_ALIASES = {
    **STATE_ALIASES,
    "D & N Haveli": "D & N Haveli",
    "Daman & Diu": "Daman & Diu",
}
DISTRICT_ALIASES_BY_LENGTH = sorted(DISTRICT_STATE_ALIASES, key=len, reverse=True)

HIMACHAL_DISTRICTS = {
    "Bilaspur",
    "Chamba",
    "Hamirpur",
    "Kangra",
    "Kinnaur",
    "Kullu",
    "Lahul And Spiti",
    "Mandi",
    "Shimla",
    "Sirmaur",
    "Solan",
    "Una",
}

GUJARAT_DISTRICTS = {
    "Ahmadabad",
    "Ahmadabad MC#",
    "Amreli",
    "Anand",
    "Banas kantha",
    "Bharuch",
    "Bhavnagar",
    "Botad",
    "Chhota Udaipur",
    "Dang",
    "Devbhoomi Dwarka",
    "Gandhinagar",
    "Gir Somnath",
    "Jamnagar",
    "Junagadh",
    "Kachchh",
    "Kheda",
    "Mahesana",
    "Mahisagar",
    "Morbi",
    "Narmada",
    "Navsari",
    "Panch Mahals",
    "Patan",
    "Porbandar",
    "Rajkot",
    "Sabar kantha",
    "Surat",
    "Surat MC#",
    "Surendranagar",
    "Tapi",
    "Vadodara",
    "Vadodara MC#",
    "Valsad",
}


@dataclass
class ExtractionResult:
    name: str
    path: Path
    rows: int
    notes: list[str]


def pdf_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing PDF: {path}")
    return subprocess.check_output(
        ["pdftotext", "-layout", str(path), "-"],
        text=True,
    )


def clean_space(value: str) -> str:
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def strip_year_prefix(text: str, current_year: int) -> tuple[str, int]:
    stripped = text.strip()
    match = re.match(r"^(20\d{2})(?:\s+Till)?(?:\s+Apr)?\s*(.*)$", stripped)
    if match:
        current_year = int(match.group(1))
        stripped = match.group(2)
    stripped = re.sub(r"^(Till|Apr)\b", "", stripped).strip()
    return stripped, current_year


def split_state_and_district(text: str, current_state: str | None) -> tuple[str | None, str]:
    stripped = clean_space(text)

    if stripped == "Islands" and current_state == "Andaman And Nicobar Islands":
        return current_state, ""

    for alias in ALIASES_BY_LENGTH:
        canonical = STATE_ALIASES[alias]
        if stripped == alias:
            return canonical, ""
        if stripped.startswith(alias + " "):
            return canonical, clean_space(stripped[len(alias) :])

    return current_state, stripped


def remove_state_prefix(text: str) -> tuple[str | None, str]:
    stripped = clean_space(text)

    for alias in DISTRICT_ALIASES_BY_LENGTH:
        canonical = DISTRICT_STATE_ALIASES[alias]
        if stripped == alias:
            return canonical, ""
        if stripped.startswith(alias + " "):
            return canonical, clean_space(stripped[len(alias) :])

    if stripped.startswith("Islands "):
        return "Andaman And Nicobar Islands", clean_space(stripped[len("Islands ") :])

    return None, stripped


def split_year_label(text: str) -> tuple[int | None, str]:
    stripped = clean_space(text)
    match = re.match(r"^(20\d{2})(?:\s+Till)?(?:\s+Apr)?\s*(.*)$", stripped)
    if match:
        return int(match.group(1)), clean_space(match.group(2))
    if stripped in {"Till", "Apr"}:
        return None, ""
    return None, stripped


def assign_centered_blocks(
    items: list[dict[str, object]],
    events: list[tuple[float, object]],
    field: str,
    default_value: object | None = None,
) -> int:
    """Assign merged-cell labels to rows using label centers.

    Excel places the label visually at the center of a merged block. In the
    text export that label often appears halfway through the rows it names.
    """

    if not items:
        return 0

    filtered: list[tuple[float, object]] = []
    for center, value in sorted(events, key=lambda event: event[0]):
        if filtered and filtered[-1][1] == value and center - filtered[-1][0] <= 1.5:
            previous_center, previous_value = filtered[-1]
            filtered[-1] = ((previous_center + center) / 2, previous_value)
        else:
            filtered.append((center, value))

    start = 0
    assigned = 0
    for center, value in filtered:
        if center < start:
            continue
        end = int(round(start + (2 * (center - start))))
        end = max(start + 1, min(len(items), end))
        for row in items[start:end]:
            row[field] = value
        assigned += end - start
        start = end

    if start < len(items):
        tail_value = filtered[-1][1] if filtered else default_value
        for row in items[start:]:
            row[field] = tail_value
        assigned += len(items) - start

    if default_value is not None:
        for row in items:
            if row.get(field) in {"", None}:
                row[field] = default_value

    return assigned


def apply_known_district_corrections(rows: list[dict[str, object]]) -> int:
    corrections = 0

    for row in rows:
        state = str(row["state"])
        district = str(row["district"])
        corrected_state = state

        if state in {"Haryana", "Jammu And Kashmir"} and district in HIMACHAL_DISTRICTS:
            corrected_state = "Himachal Pradesh"
        elif state == "Haryana" and district in GUJARAT_DISTRICTS:
            corrected_state = "Gujarat"

        if corrected_state != state:
            row["state"] = corrected_state
            corrections += 1

    return corrections


def parse_district_pdf() -> ExtractionResult:
    text = pdf_text(DISTRICT_PDF)
    rows: list[dict[str, object]] = []
    year_events: list[tuple[float, int]] = []
    state_events: list[tuple[float, str]] = []
    pending_fragments: list[str] = []
    skipped_numeric = 0

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        if "District wise" in raw_line or "Total Cases" in raw_line:
            continue

        line = raw_line.replace("\f", "")
        standalone_year = re.fullmatch(r"\s*(20\d{2})(?:\s+Till)?(?:\s+Apr)?\s*", line)
        if standalone_year:
            year_events.append((float(len(rows)), int(standalone_year.group(1))))
            continue

        numeric = re.match(r"^(?P<label>.*?)(?P<cases>\d+)\s+(?P<deaths>\d+)\s*$", line)
        if not numeric:
            year, fragment = split_year_label(line)
            if year is not None:
                year_events.append((float(len(rows)), year))
            state, remainder = remove_state_prefix(fragment)
            if state:
                state_events.append((float(len(rows)), state))
            if remainder and not re.fullmatch(r"Year|State|District", remainder):
                pending_fragments.append(remainder)
            continue

        row_index = len(rows)
        year, label = split_year_label(numeric.group("label"))
        if year is not None:
            year_events.append((row_index + 0.5, year))

        state, district = remove_state_prefix(label)
        if state:
            state_events.append((row_index + 0.5, state))

        if pending_fragments:
            district = clean_space(" ".join(pending_fragments + [district]))
            pending_fragments = []

        if not district:
            skipped_numeric += 1
            continue

        rows.append(
            {
                "year": "",
                "state": "",
                "district": district,
                "total_cases": int(numeric.group("cases")),
                "total_deaths": int(numeric.group("deaths")),
            }
        )

    assign_centered_blocks(rows, year_events, "year", 2000)
    states_assigned = 0
    for year in sorted({int(row["year"]) for row in rows}):
        start = next(index for index, row in enumerate(rows) if int(row["year"]) == year)
        year_rows = [row for row in rows if int(row["year"]) == year]
        year_events_for_state = [
            (center - start, state)
            for center, state in state_events
            if start <= center < start + len(year_rows)
        ]
        states_assigned += assign_centered_blocks(year_rows, year_events_for_state, "state")

    district_corrections = apply_known_district_corrections(rows)

    out_path = OUT_DIR / "district_malaria_2000_2024_from_pdf.csv"
    write_csv(
        out_path,
        ["year", "state", "district", "total_cases", "total_deaths"],
        rows,
    )

    years = sorted({row["year"] for row in rows})
    states = sorted({row["state"] for row in rows})
    notes = [
        f"years={years[0]}-{years[-1]}",
        f"states_or_uts={len(states)}",
        f"skipped_numeric_rows={skipped_numeric}",
        f"district_state_corrections={district_corrections}",
    ]
    return ExtractionResult("district", out_path, len(rows), notes)


def parse_state_situation_pdf() -> ExtractionResult:
    text = pdf_text(STATE_SITUATION_PDF)
    rows: list[dict[str, object]] = []
    pending_name: list[str] = []

    row_re = re.compile(r"^\s*(?:(?P<sn>\d+)\s+)?(?P<name>.*?)\s+(?P<numbers>(?:\d+\s+){19}\d+)\s*$")
    for raw_line in text.splitlines():
        line = raw_line.replace("\f", "")
        if not line.strip():
            continue
        if any(token in line for token in ["MALARIA SITUATION", "States/UTs", "Tested", "Positive"]):
            continue
        if re.search(r"\b2021\b.*\b2022\b.*\b2023\b.*\b2024\b.*\b2025\b", line):
            continue

        match = row_re.match(line)
        if not match:
            fragment = clean_space(line)
            if fragment:
                pending_name.append(fragment)
            continue

        name = clean_space(" ".join(pending_name + [match.group("name")]))
        pending_name = []
        values = [int(value) for value in match.group("numbers").split()]
        if len(values) != 20:
            continue

        for index, year in enumerate(YEARS):
            offset = index * 4
            rows.append(
                {
                    "year": year,
                    "state": name,
                    "tested": values[offset],
                    "positive": values[offset + 1],
                    "pf": values[offset + 2],
                    "deaths": values[offset + 3],
                    "source_status": "provisional" if year == 2025 else "reported",
                }
            )

    out_path = OUT_DIR / "state_malaria_situation_2021_2025_from_pdf.csv"
    write_csv(
        out_path,
        ["year", "state", "tested", "positive", "pf", "deaths", "source_status"],
        rows,
    )
    notes = [
        f"state_year_rows={len(rows)}",
        f"states_or_uts={len({row['state'] for row in rows})}",
    ]
    return ExtractionResult("state_situation", out_path, len(rows), notes)


def parse_epidemiological_pdf() -> ExtractionResult:
    text = pdf_text(EPIDEMIOLOGICAL_PDF)
    rows: list[dict[str, object]] = []
    pending_state: list[str] = []
    current_state = ""
    canonical_states = set(STATE_ALIASES.values()) | {"India"}

    data_re = re.compile(r"^\s*(?:(?P<sn>\d+)\s+)?(?P<head>.*?)(?P<year>2024|2025)\s+(?P<rest>.+)$")
    for raw_line in text.splitlines():
        line = raw_line.replace("\f", "")
        if not line.strip():
            continue
        if any(
            token in line
            for token in [
                "NATIONAL CENTRE",
                "Epidemiological Report",
                "%Change",
                "State",
                "SN",
                "Death",
                "Tested Positive",
            ]
        ):
            continue

        match = data_re.match(line)
        if not match:
            fragment = clean_space(line)
            if fragment and current_state and current_state not in canonical_states:
                current_state = clean_space(f"{current_state} {fragment}")
            elif fragment and not fragment.startswith("India"):
                pending_state.append(fragment)
            continue

        head = clean_space(match.group("head"))
        if head:
            current_state = clean_space(" ".join(pending_state + [head]))
            pending_state = []
        elif pending_state and not current_state:
            current_state = clean_space(" ".join(pending_state))
            pending_state = []

        if not current_state:
            continue

        rest = match.group("rest")
        category_match = re.search(r"(Category\s+[IVX]+)\s*$", rest)
        category = category_match.group(1) if category_match else ""
        if category:
            rest = rest[: category_match.start()]

        numbers = re.findall(r"-?\d+(?:\.\d+)?", rest)
        if len(numbers) < 7:
            continue

        year = int(match.group("year"))
        deaths = imported_cases = indigenous_cases = ""
        if year == 2025 and len(numbers) >= 13:
            deaths = int(float(numbers[-4]))
            imported_cases = int(float(numbers[-2]))
            indigenous_cases = int(float(numbers[-1]))
        elif year == 2025 and len(numbers) >= 12:
            deaths = int(float(numbers[-3]))
            imported_cases = int(float(numbers[-2]))
            indigenous_cases = int(float(numbers[-1]))

        rows.append(
            {
                "year": year,
                "state": current_state,
                "tested": int(float(numbers[0])),
                "positive": int(float(numbers[1])),
                "pf": int(float(numbers[2])),
                "pf_percent": float(numbers[3]),
                "tpr": float(numbers[4]),
                "tfr": float(numbers[5]),
                "deaths": deaths,
                "imported_cases": imported_cases,
                "indigenous_cases": indigenous_cases,
                "category": category,
            }
        )

    for index, row in enumerate(rows[:-1]):
        next_row = rows[index + 1]
        if (
            row["year"] == 2024
            and next_row["year"] == 2025
            and str(row["state"]) not in canonical_states
            and str(next_row["state"]) in canonical_states
        ):
            row["state"] = next_row["state"]

    out_path = OUT_DIR / "state_epidemiological_2024_2025_from_pdf.csv"
    write_csv(
        out_path,
        [
            "year",
            "state",
            "tested",
            "positive",
            "pf",
            "pf_percent",
            "tpr",
            "tfr",
            "deaths",
            "imported_cases",
            "indigenous_cases",
            "category",
        ],
        rows,
    )
    category_counts = Counter(row["category"] or "missing" for row in rows)
    notes = [
        f"state_year_rows={len(rows)}",
        "categories=" + ", ".join(f"{key}:{value}" for key, value in sorted(category_counts.items())),
    ]
    return ExtractionResult("epidemiological", out_path, len(rows), notes)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [
        parse_district_pdf(),
        parse_state_situation_pdf(),
        parse_epidemiological_pdf(),
    ]

    report_path = OUT_DIR / "extraction_report.txt"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("Malaria PDF extraction report\n")
        handle.write("=============================\n\n")
        for result in results:
            handle.write(f"{result.name}\n")
            handle.write(f"  file: {result.path}\n")
            handle.write(f"  rows: {result.rows}\n")
            for note in result.notes:
                handle.write(f"  {note}\n")
            handle.write("\n")

    print(f"Wrote report: {report_path}")
    for result in results:
        print(f"Wrote {result.rows} rows: {result.path}")


if __name__ == "__main__":
    main()
