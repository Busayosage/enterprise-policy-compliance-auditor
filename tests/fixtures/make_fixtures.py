"""
Generate minimal valid PDF fixture files used by the test suite.
Run once: python tests/fixtures/make_fixtures.py
Requires only the stdlib + a minimal PDF writer (no fpdf2 dependency).
Each PDF is hand-crafted using raw PDF syntax so the suite has zero
extra dependencies beyond what requirements.txt already installs.
"""
import struct
import zlib
import os

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_pdf(pages: list[tuple[str, str]]) -> bytes:
    """Build a minimal single/multi-page PDF from (title, body) pairs.
    Returns raw bytes suitable for writing to a .pdf file.
    pages: list of (page_title, body_text) — each becomes one PDF page.
    """
    def _encode(text: str) -> bytes:
        # Escape special PDF string chars
        return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").encode("latin-1", errors="replace")

    objects = []  # list of (object_id, bytes_content)
    obj_id = [0]

    def new_obj(content: bytes) -> int:
        obj_id[0] += 1
        objects.append((obj_id[0], content))
        return obj_id[0]

    # Font
    font_id = new_obj(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    page_ids = []
    for title, body in pages:
        lines = [title] + [""] + body.split("\n")
        stream_parts = [b"BT", b"/F1 11 Tf", b"50 780 Td", b"14 TL"]
        for line in lines:
            stream_parts.append(b"(" + _encode(line) + b") Tj T*")
        stream_parts.append(b"ET")
        stream = b"\n".join(stream_parts)

        stream_id = new_obj(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream"
        )
        page_id = new_obj(
            b"<< /Type /Page /MediaBox [0 0 595 842]"
            b" /Resources << /Font << /F1 " + str(font_id).encode() + b" 0 R >> >>"
            b" /Contents " + str(stream_id).encode() + b" 0 R >>"
        )
        page_ids.append(page_id)

    kids = b" ".join(str(pid).encode() + b" 0 R" for pid in page_ids)
    pages_id = new_obj(
        b"<< /Type /Pages /Kids [" + kids + b"] /Count "
        + str(len(page_ids)).encode() + b" >>"
    )

    # Patch each page to point to the Pages object
    for i, (oid, content) in enumerate(objects):
        if b"/Type /Page " in content:
            objects[i] = (oid, content + b" /Parent " + str(pages_id).encode() + b" 0 R")

    catalog_id = new_obj(
        b"<< /Type /Catalog /Pages " + str(pages_id).encode() + b" 0 R >>"
    )

    # Build PDF body
    body_parts = [b"%PDF-1.4\n"]
    offsets = {}
    for oid, content in objects:
        offsets[oid] = len(b"".join(body_parts))
        body_parts.append(
            str(oid).encode() + b" 0 obj\n" + content + b"\nendobj\n"
        )

    xref_pos = len(b"".join(body_parts))
    body_parts.append(b"xref\n0 " + str(len(objects) + 1).encode() + b"\n")
    body_parts.append(b"0000000000 65535 f \n")
    for oid in range(1, len(objects) + 1):
        body_parts.append(str(offsets[oid]).zfill(10).encode() + b" 00000 n \n")

    body_parts.append(
        b"trailer\n<< /Size " + str(len(objects) + 1).encode()
        + b" /Root " + str(catalog_id).encode() + b" 0 R >>\n"
        + b"startxref\n" + str(xref_pos).encode() + b"\n%%EOF\n"
    )
    return b"".join(body_parts)


# ---------------------------------------------------------------------------
# Fixture 1: IT Security Policy (single-document tests)
# ---------------------------------------------------------------------------
IT_POLICY_PAGES = [
    (
        "IT Security Policy v2.1",
        "Section 1 — Password Requirements\n"
        "All employees must use passwords of at least 12 characters.\n"
        "Passwords must include uppercase, lowercase, numbers and symbols.\n"
        "Passwords must be changed every 90 days.\n"
        "Sharing passwords is strictly prohibited and subject to immediate dismissal.\n"
        "\n"
        "Section 2 — Multi-Factor Authentication\n"
        "MFA is mandatory for all remote access to corporate systems.\n"
        "Approved MFA methods: hardware token, authenticator app, SMS (last resort).\n"
        "Employees who lose their MFA device must report to IT Security within 1 hour.\n",
    ),
    (
        "IT Security Policy v2.1 — Page 2",
        "Section 3 — Acceptable Use\n"
        "Corporate devices must not be used for personal cryptocurrency mining.\n"
        "Installation of unauthorised software is prohibited.\n"
        "All software must be approved by the IT Security team before installation.\n"
        "\n"
        "Section 4 — Incident Response\n"
        "Security incidents must be reported to security@company.com within 2 hours.\n"
        "The incident response team will acknowledge within 30 minutes.\n"
        "All incidents are logged in the Security Incident Register.\n",
    ),
    (
        "IT Security Policy v2.1 — Page 3",
        "Section 5 — Data Classification\n"
        "Data is classified into four tiers: Public, Internal, Confidential, Restricted.\n"
        "Restricted data must be encrypted at rest using AES-256.\n"
        "Confidential data must not be sent via unencrypted email.\n"
        "\n"
        "Section 6 — Remote Work\n"
        "Remote employees must connect via the approved corporate VPN.\n"
        "Public Wi-Fi networks must not be used without VPN.\n"
        "Screen lock must activate after 5 minutes of inactivity.\n",
    ),
]

# ---------------------------------------------------------------------------
# Fixture 2: HR Leave Policy (multi-document and cross-doc tests)
# ---------------------------------------------------------------------------
HR_POLICY_PAGES = [
    (
        "HR Leave and Absence Policy v1.3",
        "Section 1 — Annual Leave Entitlement\n"
        "Full-time employees are entitled to 25 days annual leave per year.\n"
        "Part-time employees receive leave on a pro-rata basis.\n"
        "Leave must be approved by the direct manager at least 2 weeks in advance.\n"
        "Unused leave may be carried over up to a maximum of 5 days per year.\n"
        "\n"
        "Section 2 — Sick Leave\n"
        "Employees are entitled to 10 days paid sick leave per year.\n"
        "A doctor's certificate is required for absences of 3 or more consecutive days.\n"
        "Sick leave does not carry over to the following year.\n",
    ),
    (
        "HR Leave and Absence Policy v1.3 — Page 2",
        "Section 3 — Parental Leave\n"
        "Primary caregivers are entitled to 26 weeks of parental leave.\n"
        "Secondary caregivers are entitled to 4 weeks of parental leave.\n"
        "Parental leave must be requested at least 8 weeks before the expected start date.\n"
        "\n"
        "Section 4 — Disciplinary Absence\n"
        "Unauthorised absence of more than 3 consecutive days may result in dismissal.\n"
        "Employees must notify their manager before 9am on any day of unplanned absence.\n"
        "Repeated unplanned absences will trigger a formal attendance review.\n",
    ),
]

# ---------------------------------------------------------------------------
# Fixture 3: Finance Expense Policy (cross-document tests)
# ---------------------------------------------------------------------------
FINANCE_POLICY_PAGES = [
    (
        "Finance Expense Reimbursement Policy v3.0",
        "Section 1 — Eligible Expenses\n"
        "Business travel, client meals, and approved software subscriptions are reimbursable.\n"
        "Personal expenses are not eligible for reimbursement under any circumstances.\n"
        "\n"
        "Section 2 — Approval Thresholds\n"
        "Expenses under 50 GBP require manager approval only.\n"
        "Expenses between 50 GBP and 500 GBP require manager and Finance approval.\n"
        "Expenses above 500 GBP require Director-level sign-off.\n"
        "\n"
        "Section 3 — Submission Deadline\n"
        "Expenses must be submitted within 30 days of being incurred.\n"
        "Late submissions will not be reimbursed without written Director approval.\n",
    ),
    (
        "Finance Expense Reimbursement Policy v3.0 — Page 2",
        "Section 4 — Meal Allowances\n"
        "Breakfast allowance: 15 GBP. Lunch allowance: 20 GBP. Dinner allowance: 35 GBP.\n"
        "Alcohol is not reimbursable as part of a meal expense.\n"
        "\n"
        "Section 5 — Travel Booking\n"
        "All flights must be booked through the approved travel portal.\n"
        "Economy class is required for flights under 4 hours.\n"
        "Business class requires VP-level or above pre-approval.\n"
        "\n"
        "Section 6 — Conflict with IT Policy\n"
        "Software subscriptions approved under this policy still require IT Security sign-off\n"
        "as described in the IT Security Policy Section 3.\n",
    ),
]


def write_fixture(filename: str, pages: list) -> str:
    path = os.path.join(FIXTURES_DIR, filename)
    pdf_bytes = _make_pdf(pages)
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    print(f"Written: {path} ({len(pdf_bytes):,} bytes, {len(pages)} page(s))")
    return path


if __name__ == "__main__":
    write_fixture("it_security_policy.pdf", IT_POLICY_PAGES)
    write_fixture("hr_leave_policy.pdf", HR_POLICY_PAGES)
    write_fixture("finance_expense_policy.pdf", FINANCE_POLICY_PAGES)
    print("All fixtures written.")
