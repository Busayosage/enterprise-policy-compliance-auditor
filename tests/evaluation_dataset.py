"""
Reference evaluation dataset for the Enterprise Policy Compliance Auditor.
40 questions across five categories.  All facts are sourced from the three
fixture PDFs in tests/fixtures/.

Each entry:
  id           — unique identifier
  category     — question type
  question     — the query string sent to the RAG system
  expected_source_files  — which fixture PDF(s) should be cited
  expected_page_hint     — 0-indexed page within that PDF (approximate)
  key_terms    — terms that a correct answer should contain (case-insensitive)
  should_refuse — True if the correct behaviour is a refusal (no evidence)
  notes        — explanation of the expected answer
"""

EVALUATION_QUESTIONS = [

    # ------------------------------------------------------------------
    # Category A: Direct factual (single-section, single-document)
    # ------------------------------------------------------------------
    {
        "id": "A01",
        "category": "direct_factual",
        "question": "What is the minimum password length required for all employees?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["12", "characters"],
        "should_refuse": False,
        "notes": "IT Security Policy Section 1: at least 12 characters.",
    },
    {
        "id": "A02",
        "category": "direct_factual",
        "question": "How often must employees change their passwords?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["90 days"],
        "should_refuse": False,
        "notes": "IT Security Policy Section 1: every 90 days.",
    },
    {
        "id": "A03",
        "category": "direct_factual",
        "question": "Is sharing passwords permitted under the IT Security Policy?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["prohibited", "dismissal"],
        "should_refuse": False,
        "notes": "Strictly prohibited and subject to immediate dismissal.",
    },
    {
        "id": "A04",
        "category": "direct_factual",
        "question": "How many days of annual leave are full-time employees entitled to?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["25"],
        "should_refuse": False,
        "notes": "HR Leave Policy Section 1: 25 days.",
    },
    {
        "id": "A05",
        "category": "direct_factual",
        "question": "How many days of paid sick leave are employees entitled to per year?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["10"],
        "should_refuse": False,
        "notes": "HR Leave Policy Section 2: 10 days.",
    },
    {
        "id": "A06",
        "category": "direct_factual",
        "question": "What is the dinner meal allowance for business travel?",
        "expected_source_files": ["finance_expense_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["35"],
        "should_refuse": False,
        "notes": "Finance Policy Section 4: dinner 35 GBP.",
    },
    {
        "id": "A07",
        "category": "direct_factual",
        "question": "Within how many days must expenses be submitted for reimbursement?",
        "expected_source_files": ["finance_expense_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["30"],
        "should_refuse": False,
        "notes": "Finance Policy Section 3: 30 days.",
    },
    {
        "id": "A08",
        "category": "direct_factual",
        "question": "What encryption standard is required for Restricted data at rest?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 2,
        "key_terms": ["AES-256"],
        "should_refuse": False,
        "notes": "IT Security Policy Section 5: AES-256.",
    },
    {
        "id": "A09",
        "category": "direct_factual",
        "question": "How long must the screen lock activate after inactivity for remote workers?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 2,
        "key_terms": ["5 minutes"],
        "should_refuse": False,
        "notes": "IT Security Policy Section 6: 5 minutes.",
    },
    {
        "id": "A10",
        "category": "direct_factual",
        "question": "What is the maximum number of annual leave days that can be carried over?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["5"],
        "should_refuse": False,
        "notes": "HR Leave Policy Section 1: up to 5 days.",
    },

    # ------------------------------------------------------------------
    # Category B: Multi-section (single document, multiple sections needed)
    # ------------------------------------------------------------------
    {
        "id": "B01",
        "category": "multi_section",
        "question": "What are the password composition requirements and how frequently must they be changed?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["12", "90 days", "uppercase", "lowercase"],
        "should_refuse": False,
        "notes": "Requires both Section 1 length rules and frequency rule.",
    },
    {
        "id": "B02",
        "category": "multi_section",
        "question": "What MFA methods are approved and what must an employee do if they lose their MFA device?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["hardware token", "authenticator", "1 hour"],
        "should_refuse": False,
        "notes": "Section 2: approved methods + loss reporting procedure.",
    },
    {
        "id": "B03",
        "category": "multi_section",
        "question": "Describe the full expense approval process for a 600 GBP claim.",
        "expected_source_files": ["finance_expense_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["Director", "500"],
        "should_refuse": False,
        "notes": "Over 500 GBP: Director-level sign-off required.",
    },
    {
        "id": "B04",
        "category": "multi_section",
        "question": "What are the sick leave entitlement and what documentation is required for a 4-day absence?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["10", "doctor", "certificate", "3"],
        "should_refuse": False,
        "notes": "10 days entitlement; certificate required for 3+ days.",
    },
    {
        "id": "B05",
        "category": "multi_section",
        "question": "What are the data classification tiers and what handling rule applies to Confidential data?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 2,
        "key_terms": ["Public", "Internal", "Confidential", "Restricted", "email"],
        "should_refuse": False,
        "notes": "Section 5: four tiers; Confidential not via unencrypted email.",
    },
    {
        "id": "B06",
        "category": "multi_section",
        "question": "What is the parental leave entitlement for both primary and secondary caregivers?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["26", "4"],
        "should_refuse": False,
        "notes": "26 weeks primary, 4 weeks secondary.",
    },
    {
        "id": "B07",
        "category": "multi_section",
        "question": "What are the meal allowances for breakfast, lunch, and dinner?",
        "expected_source_files": ["finance_expense_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["15", "20", "35"],
        "should_refuse": False,
        "notes": "Section 4: breakfast 15, lunch 20, dinner 35 GBP.",
    },
    {
        "id": "B08",
        "category": "multi_section",
        "question": "What must an employee do to report a security incident, and what response time can they expect?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["security@company.com", "2 hours", "30 minutes"],
        "should_refuse": False,
        "notes": "Section 4: report within 2h, acknowledge within 30m.",
    },

    # ------------------------------------------------------------------
    # Category C: Cross-document (answer requires citing two policies)
    # ------------------------------------------------------------------
    {
        "id": "C01",
        "category": "cross_document",
        "question": "Does a software subscription require both IT Security and Finance approval?",
        "expected_source_files": ["it_security_policy.pdf", "finance_expense_policy.pdf"],
        "expected_page_hint": None,
        "key_terms": ["IT Security", "Finance", "approval"],
        "should_refuse": False,
        "notes": "Finance Section 6 cross-references IT Security Section 3.",
    },
    {
        "id": "C02",
        "category": "cross_document",
        "question": "What leave must an employee take if they are the primary carer for a newborn, and how far in advance must it be requested?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["26", "8 weeks"],
        "should_refuse": False,
        "notes": "HR Policy Section 3: 26 weeks, request 8 weeks before.",
    },
    {
        "id": "C03",
        "category": "cross_document",
        "question": "What policies govern an employee's use of a corporate laptop while working remotely from a café?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 2,
        "key_terms": ["VPN", "public Wi-Fi", "screen lock"],
        "should_refuse": False,
        "notes": "IT Security Sections 3 and 6 both apply.",
    },
    {
        "id": "C04",
        "category": "cross_document",
        "question": "If an employee incurs a 450 GBP business travel expense and purchases software, what approvals are needed?",
        "expected_source_files": ["finance_expense_policy.pdf", "it_security_policy.pdf"],
        "expected_page_hint": None,
        "key_terms": ["Finance", "IT Security", "manager"],
        "should_refuse": False,
        "notes": "Finance threshold + IT Security Section 3 sign-off.",
    },

    # ------------------------------------------------------------------
    # Category D: Unsupported questions (correct behaviour = refusal)
    # ------------------------------------------------------------------
    {
        "id": "D01",
        "category": "unsupported",
        "question": "What is the company's policy on working from the moon?",
        "expected_source_files": [],
        "expected_page_hint": None,
        "key_terms": [],
        "should_refuse": True,
        "notes": "No such policy exists in any fixture document.",
    },
    {
        "id": "D02",
        "category": "unsupported",
        "question": "What is the pension contribution rate for employees?",
        "expected_source_files": [],
        "expected_page_hint": None,
        "key_terms": [],
        "should_refuse": True,
        "notes": "Pensions not covered in any fixture.",
    },
    {
        "id": "D03",
        "category": "unsupported",
        "question": "What is the process for applying for a company car?",
        "expected_source_files": [],
        "expected_page_hint": None,
        "key_terms": [],
        "should_refuse": True,
        "notes": "Company cars not covered.",
    },
    {
        "id": "D04",
        "category": "unsupported",
        "question": "What is the policy on social media use during work hours?",
        "expected_source_files": [],
        "expected_page_hint": None,
        "key_terms": [],
        "should_refuse": True,
        "notes": "Social media not mentioned in any fixture.",
    },
    {
        "id": "D05",
        "category": "unsupported",
        "question": "How many days notice must an employee give before resigning?",
        "expected_source_files": [],
        "expected_page_hint": None,
        "key_terms": [],
        "should_refuse": True,
        "notes": "Notice periods not covered.",
    },
    {
        "id": "D06",
        "category": "unsupported",
        "question": "What is the maternity pay rate?",
        "expected_source_files": [],
        "expected_page_hint": None,
        "key_terms": [],
        "should_refuse": True,
        "notes": "Pay rates not in the HR Leave Policy.",
    },

    # ------------------------------------------------------------------
    # Category E: Conflicting / edge-case questions
    # ------------------------------------------------------------------
    {
        "id": "E01",
        "category": "conflicting_or_edge",
        "question": "Can an employee use SMS for MFA, or is it not permitted?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["SMS", "last resort"],
        "should_refuse": False,
        "notes": "SMS is permitted but listed as 'last resort' — nuanced answer expected.",
    },
    {
        "id": "E02",
        "category": "conflicting_or_edge",
        "question": "Is alcohol reimbursable if included as part of a client dinner?",
        "expected_source_files": ["finance_expense_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["alcohol", "not reimbursable"],
        "should_refuse": False,
        "notes": "Section 4 explicitly states alcohol is not reimbursable.",
    },
    {
        "id": "E03",
        "category": "conflicting_or_edge",
        "question": "Can a Director fly business class on a 3-hour flight?",
        "expected_source_files": ["finance_expense_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["economy", "4 hours", "VP"],
        "should_refuse": False,
        "notes": "Economy required under 4h; business class needs VP or above.",
    },
    {
        "id": "E04",
        "category": "conflicting_or_edge",
        "question": "What happens if an employee submits an expense claim 45 days after it was incurred?",
        "expected_source_files": ["finance_expense_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["30 days", "Director", "late"],
        "should_refuse": False,
        "notes": "Late — not reimbursed without Director written approval.",
    },
    {
        "id": "E05",
        "category": "conflicting_or_edge",
        "question": "What are the consequences of taking unauthorised absence for 4 consecutive days?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["dismissal", "3"],
        "should_refuse": False,
        "notes": "Section 4: 3+ consecutive days may result in dismissal.",
    },
    {
        "id": "E06",
        "category": "conflicting_or_edge",
        "question": "Does the IT Security Policy say anything about cryptocurrency?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["cryptocurrency", "prohibited"],
        "should_refuse": False,
        "notes": "Section 3: cryptocurrency mining on corporate devices is prohibited.",
    },
    {
        "id": "E07",
        "category": "conflicting_or_edge",
        "question": "Must an employee notify anyone before 9am on a day of unplanned absence?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 1,
        "key_terms": ["manager", "9am"],
        "should_refuse": False,
        "notes": "Section 4: must notify manager before 9am.",
    },
    {
        "id": "E08",
        "category": "conflicting_or_edge",
        "question": "What is the maximum carry-over of sick leave to the following year?",
        "expected_source_files": ["hr_leave_policy.pdf"],
        "expected_page_hint": 0,
        "key_terms": ["not carry over", "does not"],
        "should_refuse": False,
        "notes": "Section 2: sick leave does not carry over — zero.",
    },
    {
        "id": "E09",
        "category": "conflicting_or_edge",
        "question": "What is the company's policy on using public Wi-Fi without a VPN?",
        "expected_source_files": ["it_security_policy.pdf"],
        "expected_page_hint": 2,
        "key_terms": ["VPN", "must not"],
        "should_refuse": False,
        "notes": "Section 6: must not use public Wi-Fi without VPN.",
    },
    {
        "id": "E10",
        "category": "conflicting_or_edge",
        "question": "If an employee needs both IT and Finance approval for software, which policy takes precedence if they conflict?",
        "expected_source_files": ["it_security_policy.pdf", "finance_expense_policy.pdf"],
        "expected_page_hint": None,
        "key_terms": ["IT Security", "Finance", "sign-off"],
        "should_refuse": False,
        "notes": "Finance Section 6 says IT Security sign-off is still required — both must be satisfied.",
    },
]

# Convenience groupings
CATEGORIES = {
    "direct_factual": [q for q in EVALUATION_QUESTIONS if q["category"] == "direct_factual"],
    "multi_section":  [q for q in EVALUATION_QUESTIONS if q["category"] == "multi_section"],
    "cross_document": [q for q in EVALUATION_QUESTIONS if q["category"] == "cross_document"],
    "unsupported":    [q for q in EVALUATION_QUESTIONS if q["category"] == "unsupported"],
    "conflicting_or_edge": [q for q in EVALUATION_QUESTIONS if q["category"] == "conflicting_or_edge"],
}

if __name__ == "__main__":
    total = len(EVALUATION_QUESTIONS)
    print(f"Total questions: {total}")
    for cat, qs in CATEGORIES.items():
        print(f"  {cat}: {len(qs)}")
    refusals = sum(1 for q in EVALUATION_QUESTIONS if q["should_refuse"])
    print(f"  should_refuse=True: {refusals}")
