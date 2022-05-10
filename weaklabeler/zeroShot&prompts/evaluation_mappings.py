TASK_MAPPINGS = {
    # Stance Benchmark
    "arc": {
        "task": "ARC",
        "id2label": {0: "unrelated", 1: "discuss", 2: "agree", 3: "disagree"},
    },
    "argmin": {
        "task": "ArgMin",
        
        "id2label": {0: "Argument_against", 1: "Argument_for"},
    },
    "fnc1": {
        "task": "FNC-1",
        
        "id2label": {0: "unrelated", 1: "discuss", 2: "agree", 3: "disagree"},
    },
    "iac1": {
        "task": "IAC",
        
        "id2label": {0: "anti", 1: "pro", 2: "other"},
    },
    "ibmcs": {
        "task": "IBM_CLAIM_STANCE",
        
        "id2label": {0: "CON", 1: "PRO"},
    },
    "perspectrum": {
        "task": "PERSPECTRUM",
        
        "id2label": {0: "UNDERMINE", 1: "SUPPORT"},
    },
    "scd": {
        "task": "SCD",
        
        "id2label": {0: "against", 1: "for"},
    },
    "semeval2016t6": {
        "task": "SemEval2016Task6",
        
        "id2label": {0: "AGAINST", 1: "FAVOR", 2: "NONE"},
    },
    "semeval2019t7": {
        "task": "SemEval2019Task7",
        
        "id2label": {0: "support", 1: "deny", 2: "query", 3: "comment"},
    },
    "snopes": {
        "task": "Snopes",
        
        "id2label": {0: "refute", 1: "agree"},
    },
    # Others
    "covidlies": {
        "task": "covidlies",
        
        "id2label": {0: "positive", 1: "negative"},
    },
    "emergent": {
        "task": "emergent",
        
        "id2label": {0: "against", 1: "for", 2: "observing"},
    },
    "mtsd": {
        "task": "mtsd",
        
        "id2label": {0: "AGAINST", 1: "FAVOR", 2: "NONE"},
    },
    "poldeb": {
        "task": "politicalDebates",
        
        "id2label": {0: "for", 1: "against"},
    },
    "rumor": {
        "task": "rumor",
        
        "id2label": {0: "endorse", 1: "deny", 2: "question", 3: "neutral", 4: "unrelated"},
    },
    "vast": {
        "task": "vast",
        
        "id2label": {0: "con", 1: "pro", 2: "neutral"},
    },
    "wtwt": {
        "task": "wtwt",
        
        "id2label": {0: "comment", 1: "refute", 2: "support", 3: "unrelated"},
    },
}