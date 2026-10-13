from pathlib import Path

import pytest

from hr_rag.loaders import detect_heading, flatten_json, humanize_dates, load_file, table_to_lines

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


@pytest.mark.parametrize("path", sorted(RAW.iterdir()), ids=lambda p: p.name)
def test_every_file_loads_with_text(path):
    if path.name == "manifest.yaml":
        pytest.skip()
    docs = load_file(path)
    assert docs, f"{path.name} produced no documents"
    assert all(d.page_content.strip() for d in docs)
    assert all(d.metadata["file_name"] == path.name for d in docs)


def test_csv_is_one_document_per_row():
    docs = load_file(RAW / "employee_directory.csv")
    assert len(docs) == 30
    assert docs[0].metadata["employee_id"] == "APS001"
    assert "Aarav Mehta" in docs[0].page_content


def test_json_profiles_one_document_per_employee():
    docs = load_file(RAW / "employee_profiles.json")
    assert len(docs) == 8
    assert "Date Of Birth" in docs[0].page_content


def test_incomplete_json_has_blank_id():
    (doc,) = load_file(RAW / "incomplete_employee_record.json")
    assert doc.metadata["employee_id"] == ""


def test_xlsx_one_document_per_sheet_with_column_names():
    docs = load_file(RAW / "holiday_calendar.xlsx")
    assert [d.metadata["sheet_name"] for d in docs] == ["Holiday Calendar", "Rules", "Summary"]
    assert "Holiday Name: New Year's Day" in docs[0].page_content


def test_docx_table_keeps_column_headers():
    docs = load_file(RAW / "leave_policy_v2.docx")
    table = next(d for d in docs if d.metadata["section_title"] == "Leave Entitlements")
    assert "Leave Type: Annual Leave | V1: 18 Days | V2: 20 Days" in table.page_content


def test_pdf_sections_and_page_numbers_and_no_running_footer():
    docs = load_file(RAW / "Employee_Handbook.pdf")
    titles = [d.metadata["section_title"] for d in docs]
    assert "5. Leave and Time Off" in titles
    leave = docs[titles.index("5. Leave and Time Off")]
    assert leave.metadata["page_number"] == 4
    assert all("SYNTHETIC TRAINING DATASET" not in d.page_content for d in docs)


def test_txt_inline_caps_headings():
    titles = [d.metadata["section_title"] for d in load_file(RAW / "work_from_home_policy.txt")]
    assert "5. Attendance" in titles and "7. Information Security" in titles


def test_html_inline_tags_do_not_split_sentences():
    docs = load_file(RAW / "hr_announcement.html")
    assert any("Effective from 01 July 2025" in d.page_content for d in docs)


@pytest.mark.parametrize("line,expected", [
    ("3. Leave Categories", "3. Leave Categories"),
    ("10. HR Policies Overview", "10. HR Policies Overview"),
    ("5.  ATTENDANCE Employees must:", "5. Attendance"),
    ("# Purpose", "Purpose"),
])
def test_detect_heading(line, expected):
    assert detect_heading(line)[0] == expected


@pytest.mark.parametrize("line", [
    "1. Submit the form to your manager before the deadline.",
    "Employees must mark attendance.",
    "1. 1. Welcome and Purpose",
])
def test_not_headings(line):
    assert detect_heading(line) is None


def test_table_two_columns_is_key_value():
    assert table_to_lines([["Department", "Engineering"], ["Location", ""]]) == ["Department: Engineering"]


def test_flatten_json_skips_blanks():
    assert flatten_json({"a_b": {"c": "x", "d": ""}, "tags": ["p", "q"]}) == ["A B - C: x", "Tags: p, q"]


def test_iso_dates_get_month_names_but_invalid_dates_are_left_alone():
    assert humanize_dates("Date: 2025-01-14") == "Date: 2025-01-14 (14 January 2025)"
    assert humanize_dates("dob 1995-31-12") == "dob 1995-31-12"


def test_holiday_sheet_is_searchable_by_month_name():
    docs = load_file(RAW / "holiday_calendar.xlsx")
    assert "(1 January 2025)" in docs[0].page_content
