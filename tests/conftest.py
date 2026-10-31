"""Synthetic PDF fixtures built in memory with PyMuPDF (no files, no network)."""

from __future__ import annotations

import pymupdf
import pytest


def make_pdf(pages: list[str], **save_kwargs) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 800), text, fontsize=10)
    data = doc.tobytes(**save_kwargs)
    doc.close()
    return data


CIRCULAR_PAGE_1 = """Example State University
Office of the Controller of Examinations

Circular No. EX/2026/14

Subject: End Semester Examinations, November 2026

All students of B.Tech and M.Tech programmes are informed that the end semester
examinations will commence on 17 November 2026. The detailed timetable is
available on the university website.

The examination fee is Rs. 1,500 per semester. Students with backlogs must pay
an additional Rs. 300 per paper.
"""

CIRCULAR_PAGE_2 = """Eligibility

Students must have a minimum attendance of 75% in each course to be eligible to
appear for the examinations.

Important Dates

Last date for fee payment without fine: 5 November 2026.
Last date with late fine of Rs. 200: 10 November 2026.
Hall tickets can be downloaded from 12 November 2026.

Contact: exams@example.edu, Phone: 044-2345-6789
"""


@pytest.fixture
def circular_pdf() -> bytes:
    return make_pdf([CIRCULAR_PAGE_1, CIRCULAR_PAGE_2])


@pytest.fixture
def blank_pdf() -> bytes:
    return make_pdf(["", ""])


@pytest.fixture
def encrypted_pdf() -> bytes:
    return make_pdf(
        [CIRCULAR_PAGE_1],
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw="student",
        owner_pw="owner",
    )
