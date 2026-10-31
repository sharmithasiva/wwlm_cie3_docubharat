import pytest

from src.field_extractor import DATE_PATTERN, extract_fields, parse_date
from src.pdf_processor import PageText
from src.pipeline import process_pdf
from src.summarizer import summarize


@pytest.mark.parametrize(
    "text, iso",
    [
        ("17 November 2026", "2026-11-17"),
        ("17th Nov, 2026", "2026-11-17"),
        ("November 17, 2026", "2026-11-17"),
        ("17/11/2026", "2026-11-17"),  # day-first, Indian convention
        ("05.01.2027", "2027-01-05"),
        ("2026-11-17", "2026-11-17"),
    ],
)
def test_date_formats(text, iso):
    assert parse_date(DATE_PATTERN.search(f"on {text} at noon")) == iso


def test_impossible_dates_are_rejected():
    assert parse_date(DATE_PATTERN.search("31/02/2026")) is None


def _fields(text: str):
    return extract_fields([PageText(1, text)])


def test_date_labels():
    f = _fields(
        "The last date for applications is 15 January 2027.\n"
        "Classes will commence on 1 February 2027.\n"
        "Issued by the Registrar on 3 October 2026.\n"
        "Registration is open from 1 December 2026 to 20 December 2026."
    )
    labels = {d.normalized: d.label for d in f.dates}
    assert labels["2027-01-15"] == "deadline"
    assert labels["2027-02-01"] == "start"
    assert labels["2026-10-03"] == "date"  # "by" far from the date is not a deadline cue
    assert labels["2026-12-20"] == "deadline"
    assert [d.normalized for d in f.deadlines] == ["2027-01-15", "2026-12-20"]


def test_amounts_including_indian_grouping_and_rupee_sign():
    f = _fields("Fee: Rs. 1,500 per semester. Income must not exceed Rs. 2,50,000. Fine of ₹200. Deposit 3,000/-")
    assert [m.normalized for m in f.fees] == ["1500", "250000", "200", "3000"]


def test_contacts():
    f = _fields(
        "Write to help@example.gov.in or call +91 40 2345 6789.\n"
        "Toll free: 1800-000-4455. Visit www.portal.example.in for details.\n"
        "Results on 12-01-2027."
    )
    by_label = {}
    for c in f.contacts:
        by_label.setdefault(c.label, []).append(c.value)
    assert by_label["email"] == ["help@example.gov.in"]
    assert by_label["phone"] == ["+91 40 2345 6789", "1800-000-4455"]
    assert by_label["website"] == ["www.portal.example.in"]


def test_title_org_reference_from_letterhead():
    text = (
        "KAVERI STATE UNIVERSITY\nOffice of the Controller of Examinations\n"
        "Circular No. KSU/COE/2026/41\n\nSubject: End Semester Examinations, 2026\n\nBody text here."
    )
    f = _fields(text)
    assert f.title.value == "End Semester Examinations, 2026"
    assert f.organization.value == "KAVERI STATE UNIVERSITY"
    assert f.reference_number.value == "KSU/COE/2026/41"


def test_eligibility_and_actions_are_quoted_sentences():
    f = _fields(
        "Students must have a minimum attendance of 75% to be eligible. "
        "Applicants should upload the income certificate on the portal. The weather was pleasant."
    )
    assert f.eligibility[0].value.startswith("Students must have a minimum attendance")
    assert any("upload the income certificate" in a.value for a in f.action_items)
    assert all("weather" not in a.value for a in f.action_items)
    assert all(v.source == "rule" for v in f.eligibility + f.action_items)


def test_missing_fields_stay_empty():
    f = _fields("A short note without any dates, money or contacts in it at all.")
    assert f.dates == [] and f.fees == [] and f.contacts == []
    assert f.organization is None and f.reference_number is None


def test_fields_from_pdf_keep_page_numbers(circular_pdf):
    doc = process_pdf(circular_pdf, "circular.pdf")
    f = extract_fields(doc.pages)
    pages = {d.normalized: d.page for d in f.dates}
    assert pages["2026-11-17"] == 1
    assert pages["2026-11-05"] == 2
    assert {c.value for c in f.contacts} >= {"exams@example.edu", "044-2345-6789"}


def test_summary_is_extractive_and_structured(circular_pdf):
    doc = process_pdf(circular_pdf, "circular.pdf")
    summary = summarize(doc.pages)
    assert summary.purpose == "End Semester Examinations, November 2026"
    assert summary.main_points
    # Every main point is copied verbatim from the document.
    for point in summary.main_points:
        assert " ".join(point.text.split()) in " ".join(doc.full_text.split())
    # Dates are de-duplicated by calendar day.
    days = [d.normalized for d in summary.important_dates]
    assert len(days) == len(set(days))
    assert "extractive" in summary.method
