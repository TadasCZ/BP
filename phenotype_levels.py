# Dictionary defining which phenotype levels are included in the analysis.
# Note: "new/old dict" refers to the source of the classification logic.
phenotype_levels_used = {
    "AWCO_REV": [0, 11, 20, 40, 52, 61, 70, 80, 100], # new dict
    "BLPUB_VEG": [1, 2, 3], # new dict
    "PEX_REPRO": [1, 3, 5, 7, 9], # new dict
    "BLSCO_REV_VEG": [60, 80, 81, 84], # new dict
    "LIGCO_REV_VEG": [0, 11, 60, 80, 81, 84], # new dict
    "CUAN_REPRO": [1, 3, 5, 7, 9], # new dict
    "SPKF": [1, 2, 3, 4, 5], # new dict
    "LSEN": [1, 3, 5, 7, 9], # new dict
    "CCO_REV_VEG": [0, 60, 61, 80], # new dict
    "PTH": [1, 2, 3], # new dict
    "FLA_REPRO": [1, 3, 5, 7], # new dict
    "LPCO_REV_POST": [20, 42, 52, 53, 54, 80, 82, 90, 91, 100], # new dict; "10" is missing
    "SLLT_CODE": [1, 3, 5, 7, 9], # new dict
    "APCO_REV_REPRO": [10, 20, 52, 60, 70, 71, 80, 87, 100], # new dict
    "SCCO_REV": [10, 50, 51, 55, 70, 80, 88], # new dict
    "AWPR_REPRO": [0, 1, 5, 7, 9, 999], # old dict
    "CUDI_CODE_REPRO": [1, 2, 999], # old dict
    "ENDO": [1, 2, 3], # new dict
    "CUNO_CODE_REPRO": [1, 2, 3, 999], # old dict
    "LLT_CODE": [1, 2, 3, 4, 5, 999], # old dict
    "CULT_CODE_REPRO": [1, 2, 3, 4, 5, 6, 7, 999], # old dict
    "SDHT_CODE": [1, 2, 3, 999], # old dict
    "PLT_CODE_POST": [1, 2, 3, 4, 5, 999], # old dict
    "SLCO_REV": [20, 40, 70, 80], # new dict
    # 2k border
    "LIGSH": [0, 1, 2, 3], # new dict
    "LPPUB": [1, 2, 3, 4, 5], # new dict
    "SECOND_BR_REPRO": [0, 1, 2, 3] # new dict
}

# List of phenotypes excluded from the analysis.
phenotype_levels_not_used = {
    "BLCO_REV_VEG": [], # old and new dict
    "AUCO_REV_VEG": [], # old and new dict
    "INCO_REV_REPRO": [], # old and new dict
    "CUST_REPRO": [], # old and new dict
    "PTY": [], # old and new dict
    "LA": [], # old and new dict
    "PA_REPRO": [], # old and new dict
    "PSH": [], # old and new dict
    # under 1k border
    "NOCO_REV": [],
    "APCO_REV_POST": [],
    "APSH": [],
    "BLANTHPR_VEG": [],
    "FLA_EREPRO": [],
    "NOANTH": [],
    "AWCO_LREV": [],
    "BLSCO_ANTH_VEG": [],
    "INANTH": [],
    "APANTH_REPRO": [],
    "AWDIST": [],
    "BLANTHDI_VEG": []
}
