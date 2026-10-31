# Sample documents

All seven PDFs are **synthetic**. The universities, boards, the "State of Vindhya", people, reference numbers, amounts, e-mail addresses (`*.example.*` domains), phone numbers and websites are fictional. Every page carries the footer "SYNTHETIC SAMPLE DOCUMENT". They are safe to redistribute with this project.

They are generated reproducibly by:

```bash
python scripts/make_sample_documents.py
```

| File | Type | Pages | Purpose in testing |
|---|---|---|---|
| `exam_circular.pdf` | University examination circular | 2 | Dates, fees, late fine, attendance eligibility, hall tickets, contacts |
| `admission_notice.pdf` | Admission notice (M.Tech) | 2 | Category-wise fees, eligibility percentages, multi-date schedule |
| `scholarship_announcement.pdf` | Scholarship announcement | 2 | Indian number grouping (Rs. 2,50,000), income limit, documents list |
| `fee_notification.pdf` | Fee notification | 2 | Table-like fee lines, late fee per day, refund policy |
| `government_notice.pdf` | Government public notice | 1 | Age eligibility, registration window, toll-free helpline |
| `noisy_hostel_notice.pdf` | Hostel notice with poor formatting | 2 | Hard-wrapped lines, hyphenation, running headers and page numbers |
| `scanned_library_notice.pdf` | Image-only (scanned-style) notice | 1 | No text layer: the app must report that OCR is needed |
