import re

CATEGORY_RULES = {

    "Leave Policy": [
        "leave",
        "vacation",
        "annual leave",
        "casual leave",
        "sick leave",
        "maternity",
        "paternity",
        "earned leave"
    ],

    "Attendance": [
        "attendance",
        "late",
        "late coming",
        "absent",
        "working hours",
        "shift",
        "biometric",
        "clock in",
        "clock out"
    ],

    "Employee Benefits": [
        "insurance",
        "medical",
        "benefits",
        "reimbursement",
        "claim",
        "pf",
        "gratuity",
        "bonus"
    ]

}

INTENT_RULES = {

    "Greeting":[
        "hi",
        "hello",
        "good morning",
        "good evening",
        "hey"
    ],

    "Goodbye":[
        "bye",
        "goodbye",
        "see you"
    ],

    "Thanks":[
        "thanks",
        "thank you"
    ],

    "Comparison":[
        "difference",
        "compare",
        "comparison",
        "vs",
        "better"
    ],

    "Eligibility":[
        "eligible",
        "eligibility",
        "qualify",
        "who can"
    ],

    "Procedure":[
        "how",
        "process",
        "procedure",
        "steps",
        "apply"
    ],

    "Document":[
        "summary",
        "summarize",
        "document",
        "pdf"
    ],

    "Information":[
        "what",
        "when",
        "where",
        "policy",
        "rule"
    ]
}