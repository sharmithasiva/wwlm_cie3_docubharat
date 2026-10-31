"""Generate the synthetic sample PDFs in data/sample_documents/.

All organisations, people, numbers and contact details are fictional, created for
testing DocuBharat. The PDFs are reproducible: run

    python scripts/make_sample_documents.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_documents"
FOOTER = "SYNTHETIC SAMPLE DOCUMENT - fictional content created for DocuBharat testing"

DOCUMENTS: dict[str, list[str]] = {
    "exam_circular.pdf": [
        """KAVERI STATE UNIVERSITY
Office of the Controller of Examinations
Circular No. KSU/COE/2026/41                                   Date: 20 October 2026

Subject: Conduct of End Semester Examinations, November-December 2026

All Heads of Departments and students of the B.Tech, M.Tech and MBA programmes are hereby informed that the End Semester Examinations for the odd semester will commence on 24 November 2026 and conclude on 12 December 2026. The detailed subject-wise timetable will be published on the university website www.kaveri-university.example.in by 5 November 2026.

1. Examination Fee
The examination fee for regular students is Rs. 1,800 per semester. Students appearing for supplementary (backlog) papers must pay an additional Rs. 350 per paper. The fee must be paid through the online student portal only; cash and demand drafts will not be accepted.

2. Last Dates for Fee Payment
The last date for payment of the examination fee without fine is 7 November 2026. Payment with a late fine of Rs. 500 will be accepted until 14 November 2026. No payments will be accepted after 14 November 2026 under any circumstances.
""",
        """3. Eligibility to Appear
Students must have a minimum attendance of 75% in each course to be eligible to appear for the examinations. Students with attendance between 65% and 75% may apply for condonation on medical grounds with valid certificates, subject to a condonation fee of Rs. 1,000. Students with attendance below 65% are not eligible and will be detained.

4. Hall Tickets
Hall tickets will be available for download from the student portal from 18 November 2026. Students must carry the printed hall ticket and their university identity card to every examination. Students without a valid hall ticket will not be permitted to enter the examination hall.

5. Conduct of Examinations
Students should report to the examination hall 30 minutes before the start of each examination. Mobile phones, smart watches and other electronic devices are strictly prohibited inside the examination hall. Any student found using unfair means will be dealt with as per the university examination regulations.

6. Results
Results are expected to be declared in the second week of January 2027.

For queries, contact the Examination Cell at coe@kaveri-university.example.in or call 0821-2419-300 between 10:00 AM and 5:00 PM on working days.

Controller of Examinations
Kaveri State University
""",
    ],
    "admission_notice.pdf": [
        """DECCAN INSTITUTE OF TECHNOLOGY
Admissions Office, Hyderabad Campus
Notice No. DIT/ADM/2027/03

Subject: Admission to M.Tech Programmes for the Academic Year 2027-28

Applications are invited from eligible candidates for admission to the full-time M.Tech programmes in Computer Science, Data Science, VLSI Design and Structural Engineering for the academic year 2027-28. The total intake is 180 seats across all four programmes.

Eligibility
Candidates must hold a B.E. or B.Tech degree in a relevant discipline with a minimum of 60% aggregate marks or a CGPA of 6.5 on a 10-point scale. For candidates belonging to SC, ST and PwD categories, the minimum requirement is 55% aggregate marks or a CGPA of 6.0. Candidates in the final year of their degree may also apply, provided they submit proof of passing by 31 July 2027. There is no upper age limit for admission.

Application Fee
The application fee is Rs. 1,200 for General and OBC candidates and Rs. 600 for SC, ST and PwD candidates. The fee is non-refundable and must be paid online at the time of submitting the application.
""",
        """Important Dates
Online applications open on 2 January 2027.
The last date for submission of online applications is 15 February 2027.
The DIT Entrance Test (DET) will be held on 7 March 2027 at Hyderabad, Bengaluru and Chennai.
Results of the entrance test will be announced on 25 March 2027.
Counselling and seat allotment will be conducted from 10 April 2027 to 14 April 2027.

Selection Procedure
Selection will be based on the DIT Entrance Test score (70% weightage) and a personal interview (30% weightage). Candidates with a valid GATE score above 500 are exempted from the entrance test and will be called directly for the interview.

Documents Required at Counselling
Candidates must bring original mark sheets and degree certificate, a government-issued photo identity proof, category certificate (if applicable), and two recent passport-size photographs.

Tuition fee for the programme is Rs. 95,000 per year, payable at the time of admission.

How to Apply
Candidates should apply online through the admissions portal at https://admissions.dit.example.in. Applications sent by post or e-mail will not be accepted.

For assistance, contact admissions@dit.example.in or call +91 40 2345 6789.
""",
    ],
    "scholarship_announcement.pdf": [
        """VINDHYA STATE SCHOLARSHIP BOARD
Department of Higher Education, Government of Vindhya (fictional state)
Notification No. VSSB/SCH/2026-27/12

Subject: Merit-cum-Means Scholarship for Undergraduate Students, 2026-27

The Vindhya State Scholarship Board invites applications for the Merit-cum-Means Scholarship for students enrolled in the first year of undergraduate degree programmes in recognised colleges and universities in the state for the academic year 2026-27.

Scholarship Amount
Selected students will receive Rs. 25,000 per annum for the duration of the course, up to a maximum of four years. In addition, a one-time book allowance of Rs. 5,000 will be paid in the first year. The amount will be transferred directly to the student's bank account.

Eligibility Criteria
Applicants must be permanent residents of the State of Vindhya. Applicants must have scored at least 75% marks in the Class XII examination. The annual family income from all sources must not exceed Rs. 2,50,000. Students already receiving any other government scholarship are not eligible to apply.
""",
        """How to Apply
Eligible students must register and apply on the State Scholarship Portal at www.vssb.example.gov.in. The online application window opens on 1 November 2026. The last date to submit the online application is 31 December 2026. Institutes must complete verification of applications by 15 January 2027. The list of selected candidates will be published on 15 February 2027.

Documents to be Uploaded
Applicants must upload the following documents: Class XII mark sheet, income certificate issued by the Tahsildar, residence certificate, bank passbook first page, and the admission receipt from the institute.

Renewal
The scholarship will be renewed each year only if the student passes all subjects and maintains an attendance of at least 80%.

Students who submit false information will be disqualified and the scholarship amount already paid will be recovered.

Helpline: 1800-000-4455 (toll free), email: help@vssb.example.gov.in
""",
    ],
    "fee_notification.pdf": [
        """NARMADA COLLEGE OF ENGINEERING
Accounts Section
Circular No. NCE/ACC/2026/18                                   Date: 1 June 2026

Subject: Fee Structure and Payment Schedule for the Academic Year 2026-27

All students are informed that the fee structure for the academic year 2026-27 has been approved by the Governing Council. The fees for the odd semester are as follows:

Tuition Fee: Rs. 48,000 per semester
Development Fee: Rs. 6,500 per semester
Library and Laboratory Fee: Rs. 3,000 per semester
Examination Fee: Rs. 1,500 per semester
Hostel Rent (optional): Rs. 32,000 per semester
Mess Advance (optional, for hostel residents): Rs. 21,000 per semester

The total payable for a day scholar is Rs. 59,000 per semester. The total payable for a hostel resident is Rs. 1,12,000 per semester.

Payment Schedule
The last date for payment of fees for the odd semester is 31 July 2026. A late fee of Rs. 100 per day will be charged from 1 August 2026 to 20 August 2026. Students who have not paid their fees by 20 August 2026 will have their registration for the semester cancelled.
""",
        """Mode of Payment
Fees must be paid through the college payment portal or by NEFT to the college account. Students paying by NEFT must upload the transaction receipt on the student portal within three days of payment.

Fee Concession
Students whose annual family income is below Rs. 1,00,000 may apply for a 50% tuition fee concession by submitting an income certificate to the Accounts Section by 15 July 2026.

Refund Policy
Students who withdraw before 31 August 2026 will receive a refund of the tuition fee after deduction of a processing charge of Rs. 1,000. No refund will be given for withdrawals after 31 August 2026. Hostel and mess fees are refundable on a pro-rata basis.

For fee-related queries, students should contact the Accounts Section at accounts@nce.example.ac.in or 0755-2661-450.

Finance Officer
Narmada College of Engineering
""",
    ],
    "government_notice.pdf": [
        """DIRECTORATE OF DIGITAL SERVICES
Government of Vindhya (fictional state)
Public Notice No. DDS/PN/2026/77                               Date: 15 November 2026

Subject: Free Digital Literacy Training Programme for Citizens

The Directorate of Digital Services announces a free Digital Literacy Training Programme for citizens of the State of Vindhya. The programme aims to help citizens use online government services, digital payments and e-mail safely.

Programme Details
The training will be conducted at all Common Service Centres and public libraries in the state. Each batch will run for 10 days, with 2-hour sessions every day. Participants who complete the training and pass the final assessment will receive a certificate from the Directorate. There is no fee for the training.

Who Can Apply
Residents of the State of Vindhya aged between 18 and 60 years are eligible. Applicants must be able to read in English, Hindi or the regional language. Preference will be given to women, senior citizens and residents of rural areas.

Registration
Registration will be open from 1 December 2026 to 20 December 2026. Citizens can register at the nearest Common Service Centre or online at www.digital.vindhya.example.gov.in. Applicants must bring an identity proof and address proof at the time of registration. The first batches will begin on 5 January 2027.

For more information, call the helpline 1800-000-3366 or write to digitalskills@vindhya.example.gov.in.

Director, Directorate of Digital Services
""",
    ],
}

# Simulates text from a poorly produced PDF: hard-wrapped lines, hyphenated
# breaks, stray spacing and running headers/page numbers.
NOISY_DOCUMENT = [
    """Page 1 of 2
GODAVARI   UNIVERSITY
Office  of the  Dean, Student Welfare
Notice No. GU/DSW/2026/09
Subject: Hostel Allotment for the Academic Year 2026-27
Students seeking hostel accommo-
dation for the academic year 2026-27 must
apply through the online portal. The   last date
for submitting hostel applications is
25 June 2026. Allotment will be made on the basis
of distance from home and academic merit.
The hostel admission fee is Rs.  4,500 (one
time, non-refundable) and the annual room rent is
Rs. 38,000 , payable before 15 July 2026.
1
""",
    """Page 2 of 2
Students allotted rooms must report to the Chief Warden's
office with the allotment letter, two photo-
graphs and a medical fitness certificate between 20 July 2026
and 25 July 2026. Rooms not occupied by 25 July 2026 will
be allotted to wait-listed students.
For queries contact hostel@godavari.example.edu.in
2
""",
]

SCANNED_TEXT = """SAPTAGIRI UNIVERSITY
Notice: Library Timings During Examinations
The central library will remain open from 8:00 AM to 10:00 PM
from 20 November 2026 to 15 December 2026."""


def _write_pages(doc: pymupdf.Document, pages: list[str], fontsize: float = 10) -> None:
    for text in pages:
        page = doc.new_page(width=595, height=842)  # A4
        overflow = page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=fontsize, fontname="helv")
        if overflow < 0:
            raise ValueError(f"Text overflowed the page by {-overflow:.0f} points; split it across pages.")
        page.insert_text((50, 815), FOOTER, fontsize=7, color=(0.4, 0.4, 0.4))


def build_text_pdf(pages: list[str], path: Path, fontsize: float = 10) -> None:
    doc = pymupdf.open()
    _write_pages(doc, pages, fontsize)
    doc.set_metadata({"title": path.stem, "author": "DocuBharat synthetic data", "subject": FOOTER})
    doc.save(path, garbage=3, deflate=True)
    doc.close()


def build_scanned_pdf(text: str, path: Path) -> None:
    """Render text to an image and embed only the image, so the PDF has no text layer."""
    source = pymupdf.open()
    _write_pages(source, [text], fontsize=12)
    pixmap = source[0].get_pixmap(dpi=110)
    source.close()

    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(page.rect, pixmap=pixmap)
    doc.save(path, garbage=3, deflate=True)
    doc.close()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCUMENTS.items():
        build_text_pdf(pages, OUT_DIR / name)
    build_text_pdf(NOISY_DOCUMENT, OUT_DIR / "noisy_hostel_notice.pdf")
    build_scanned_pdf(SCANNED_TEXT, OUT_DIR / "scanned_library_notice.pdf")
    for path in sorted(OUT_DIR.glob("*.pdf")):
        print(f"{path.name:32s} {path.stat().st_size / 1024:6.1f} KB")


if __name__ == "__main__":
    main()
