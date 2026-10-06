# Imports
import pandas as pd
import re

"""
--------------------------------------------------
INITIAL DRUG RECOMMENDATIONS BASED ON SEIZURE TYPES
---------------------------------------------------
"""

# Start with a dictionary
INITIAL_CLASSIFICATIONS = {
    "Absence": {1: "ETS, VPA", 2: "LTG", 3: "LEV, ZNS, ACT, CLB, CLN"},
    "Myoclonic": {1: "CLN, VPA, LEV", 2: "CLB", 3: "TPM, ZNS, PB, NTR"},
    "Primary generalized tonic-clonic": {
        1: "VPA", 2: "LTG, LEV", 3: "OXC, CLB, ZNS, LCM, TPM, PB, PER, CBZ, PHT, BRV"},
    "Primary generalized tonic-clonic + myoclonic": {
        1: "VPA", 2: "LEV", 3: "CLN, LTG, ZNS, CLB, TPM, PB, PER, PHT, LCS, BRV"},
    "Primary generalized tonic-clonic + myoclonic + absence": {
        1: "VPA", 2: "LTG, LEV", 3: "TPM, ZNS, CLB, CLN"},
    "Myoclonic + absence": {
        1: "VPA, ETS", 2: "LTG, LEV", 3: "TPM, ZNS, CLB, CLN"},
    "Focal": {1: "LEV, CBZ, LTG, OXC, ESL, LCM", 2: "TPM, VPA, PER, PHT, BRV, ZNS", 3: "CLB, PB, GBP, PGB"},
    "Unknown whether focal or generalized seizure type + age 21 or older": {
        1: "LEV, CBZ, LTG, OXC, ESL, LCM", 2: "VPA, PER, PHT, BRV, CLB, TPM, ZNS", 3: "PB, GBP, PGB"},
    "Unknown whether focal or generalized seizure type + age under 21": {
        1: "VPA, LTG, LEV", 2: "CLB, CBZ, PER, ESL, OXC, LCM", 3: "PB, TPM, ZNS"},
}

# Function that turns dictionary into DataFrame
def build_hierarchy_table(raw_dict):
    """Turn INITIAL_CLASSIFICATION into a DataFrame with columns: seizure_type, group, drug."""
    rows = []
    # Iterate through (seizure type, hierarchy) key-value pairs
    for seizure_type, groups in raw_dict.items():
        # Iterate through (level, drug list) key-value pairs
        for group, drug_string in groups.items():
            # Iterate through drug list ("LTG, LEV" -> ["LTG", "LEV"])
            for item in re.split(r",\s+", drug_string):
                rows.append({"seizure_type": seizure_type, "group": group, "drug": item})

    return pd.DataFrame(rows)

HIERARCHY_DF = build_hierarchy_table(INITIAL_CLASSIFICATIONS)

"""
----------------------------
CONSTANTS + HELPER FUNCTIONS
----------------------------
"""

# List of valid seizure types sub-lists for generalized and focal/uncertain (this is helpful for certain rules)
VALID_SEIZURE_TYPES = list(INITIAL_CLASSIFICATIONS)
GENERALIZED = ["Absence", "Myoclonic", "Primary generalized tonic-clonic", "Primary generalized tonic-clonic + myoclonic", 
               "Primary generalized tonic-clonic + myoclonic + absence", "Myoclonic + absence"]
FOCAL_OR_UNCERTAIN = ["Focal", "Unknown whether focal or generalized seizure type + age 21 or older", 
                      "Unknown whether focal or generalized seizure type + age under 21"]

# Inputs that are True/false
BOOLEAN_KEYS = [
    'Pre-menopausal', 'Co-medications (non-contraceptive)', 'Oral contraceptives',
    'Brain tumor requiring chemo/radiation', 'Hepatic failure', 'Obesity',
    'Diabetes mellitus', 'Bleeding disorders', 'Neutropenia', 'Renal stones',
    'Renal failure', 'Allergy to any drug', 'Depression',
    'History of irritability or aggressive behavior', 'Migraine',
]

# All possible input keys
ALL_KEYS = ['Seizure type', 'Sex', 'Age'] + BOOLEAN_KEYS

# Function for checking input data
def validate_and_fill(input_dict):
    """Check types/values of the inputs and return a copy with defaults filled in.

    Raises TypeError for wrong types and ValueError for missing/unknown keys or bad values.
    """
    
    # Check for dictionary
    if not isinstance(input_dict, dict):
        raise TypeError(f"Input must be a dictionary, got {type(input_dict).__name__}.")

    # 'Seizure type' is the only required input
    if input_dict.get('Seizure type') is None:
        raise ValueError("'Seizure type' is required.")

    # Catch typos in key names, e.g. 'Obesety'
    # We do this by checking if there are keys in input_dict that do not appear in ALL_KEYS
    unknown_keys = set(input_dict) - set(ALL_KEYS)
    if unknown_keys:
        raise ValueError(f"Unrecognized input key(s): {sorted(unknown_keys)}. Allowed keys: {ALL_KEYS}")

    # Initialize default input data where everything is optional: None for Sex/Age, and False for every yes/no field
    filled = {'Sex': None, 'Age': None, **{key: False for key in BOOLEAN_KEYS}}
    # Copy over the user-inputted values, ignoring any set to None
    filled.update({k: v for k, v in input_dict.items() if v is not None})

    # Type checks: Seizure type and Sex must be strings, Age must be integer or float, and all others must be True/False
    if not isinstance(filled['Seizure type'], str):
        raise TypeError("'Seizure type' must be a string.")
    if filled['Sex'] is not None and not isinstance(filled['Sex'], str):
        raise TypeError("'Sex' must be a string.")
    # bool is a subclass of int in Python, so you have to exclude it explicitly 
    if filled['Age'] is not None and (isinstance(filled['Age'], bool) or not isinstance(filled['Age'], (int, float))):
        raise TypeError("'Age' must be a number (int or float).")
    for key in BOOLEAN_KEYS:
        if not isinstance(filled[key], bool):
            raise TypeError(f"'{key}' must be True or False, got {type(filled[key]).__name__}.")

    # Value checks: make sure seizure type is valid, Sex is either Male or Female, Age is non-negative and >= 10,
    # and that pre-menopausal is not set to True for males
    if filled['Seizure type'] not in VALID_SEIZURE_TYPES:
        raise ValueError(f"Unrecognized seizure type: '{filled['Seizure type']}'. Choose from: {VALID_SEIZURE_TYPES}")
    if filled['Sex'] not in (None, 'Male', 'Female'):
        raise ValueError("'Sex' must be 'Male' or 'Female'.")
    if filled['Age'] is not None and filled['Age'] < 0:
        raise ValueError("'Age' cannot be negative.")
    if filled['Age'] is not None and filled['Age'] < 10:
        raise ValueError("EpiPick is not configured for individuals whose seizures began before the age of 10.")
    if filled['Pre-menopausal'] and filled['Sex'] != 'Female':
        raise ValueError("'Pre-menopausal' can only be True when 'Sex' is 'Female'.")

    return filled

# Define the simpler rules here.
# Remember that upgrading a drug means subtracting from its level, e.g., -1, while downgrading means addition, e.g., +1
SIMPLE_RULES = {
    'Oral contraceptives': [(+1, ['PER', 'TPM']),
                            (+2, ['ESL', 'CBZ', 'OXC', 'PB', 'PHT'])],
    'Brain tumor requiring chemo/radiation': [(+2, ['ESL', 'CBZ', 'OXC', 'PB', 'PHT'])],
    'Hepatic failure': [(-1, ['GBP', 'LCM', 'LEV']),
                        (+2, ['VPA'])],
    'Diabetes mellitus': [(+1, ['CBZ', 'PER', 'PHT', 'VPA'])],
    'Neutropenia': [(+1, ['CBZ', 'PHT'])],
    'Renal stones': [(+1, ['ACT', 'TPM', 'ZNS'])],
    'Renal failure': [(-1, ['CLB', 'ETS', 'LTG', 'CBZ', 'PHT']),
                      (+1, ['VPA', 'LEV', 'ZNS', 'ACT', 'CLN', 'TPM', 'PB', 'NTR', 'OXC',
                            'LCM', 'PER', 'BRV', 'LCS', 'ESL', 'GBP', 'PGB'])],
    'Allergy to any drug': [(+1, ['ESL', 'CBZ', 'LTG', 'OXC', 'PB', 'PHT', 'ZNS'])],
    'Depression': [(-1, ['LTG']),
                   (+1, ['CLB', 'CBZ', 'LEV', 'NTR', 'PB'])],
    'History of irritability or aggressive behavior': [(+1, ['LEV', 'PER', 'PB', 'TPM'])],
    'Migraine': [(-1, ['TPM', 'VPA'])],
}

# Helper functions for locking and adjusting levels
def shift(df, drug, amount):
    """Move the given drug by 'amount' groups, if the drug is not locked."""
    row = (df['drug'] == drug) & (~df['locked'])
    df.loc[row, 'group'] += amount

def shift_multiple(df, drug_list, amount):
    """Move each non-locked drug in the drug list by 'amount' groups."""
    # Check for and remove drugs not in the DF
    drug_list = [x for x in drug_list if x in list(df['drug'])]
    # Check for and remove locked drugs
    locked_flags = [list(df[df['drug'] == x]['locked'])[0] for x in drug_list]
    drug_list = [drug_list[i] for i in range(len(drug_list)) if not locked_flags[i]]
    for drug in drug_list:
        shift(df, drug, amount)

def lock(df, drug, group):
    """Fix a drug at a specific group so no later rule can change it."""
    rows = df['drug'] == drug
    df.loc[rows, 'group'] = group
    df.loc[rows, 'locked'] = True

# Helper function for configuring the final results
def map_groups(groups, max_label=4):
    unique_sorted = sorted(set(groups))
    return {g: min(rank, max_label) for rank, g in enumerate(unique_sorted, start=1)}


"""
----------------
PRIMARY FUNCTION
----------------
"""
def epipick(input_dict, top_only=False):
    """Return EpiPick recommendations.

    Parameters
    ----------
    input_dict : dict
        Patient description (only 'Seizure type' is required)
    top_only : bool, default False
        If True, return only the most-recommended non-empty level
        (normally level 1), instead of all three levels of recommendation.

    Returns
    -------
    dict
        {1: [drugs], 2: [drugs], 3: [drugs]}, or just {1: [drugs]} if top_only=True.
    """

    # Check for bad data 
    input_dict = validate_and_fill(input_dict)
    seizure_type = input_dict['Seizure type']

    # Start from the base table for this seizure type
    df = HIERARCHY_DF[HIERARCHY_DF['seizure_type'] == seizure_type].copy()[['drug', 'group']]
    # Initialize 'locked', which tells us whether the drug is locked in position and cannot be upgraded/downgraded
    df['locked'] = False

    # Assume that all females under the age of 40 are premenopausal
    if input_dict['Sex'] == 'Female' and input_dict['Age'] < 40 and not input_dict['Pre-menopausal']:
        input_dict['Pre-menopausal'] = True

    # According to footnotes of Table 1, CLN is always locked into group 3 for certain epilepsy types.
    if (input_dict['Seizure type'] == 'Primary generalized tonic-clonic + myoclonic') or (input_dict['Seizure type'] == 'Primary generalized tonic-clonic + myoclonic + absence'):
        lock(df, 'CLN', 3)

    """
    The more complicated rules are implemented here. 
    """
    # Pre-menopausal women
    if input_dict['Sex'] == 'Female' and input_dict['Pre-menopausal']:
        if seizure_type == 'Focal':
            # Remove VPA altogether for focal seizures
            df = df[df['drug'] != 'VPA'].copy()         
        elif seizure_type == 'Primary generalized tonic-clonic':
            lock(df, 'VPA', 2)
            lock(df, 'LTG', 1)
            lock(df, 'LEV', 1)
        else:
            lock(df, 'VPA', 3)

    # Age over 65
    if input_dict['Age'] is not None and input_dict['Age'] > 65:
        shift_multiple(df, ['GBP', 'LCM', 'LEV', 'LTG'], -1)

    # Non-contraceptive co-medications + generalized epilepsy
    if input_dict['Co-medications (non-contraceptive)']:
        if seizure_type in GENERALIZED:
            shift_multiple(df, ['ETS', 'LCM', 'LEV', 'LTG', 'PER', 'ZNS'], -1)
            shift_multiple(df, ['CBZ', 'PB', 'PHT'], 1)
        else:
            shift_multiple(df, ['BRV', 'GBP', 'LCM', 'LEV', 'LTG', 'PER', 'ZNS'], -1)
            shift_multiple(df, ['CBZ', 'PB', 'PHT'], 1)
            
    # Obesity and bleeding disorders means VPA gets downgraded (number of levels depends on seizure type)
    # 1: Generalized, 2: focal/uncertain
    vpa_penalty = 1 if seizure_type in GENERALIZED else 2 
    if input_dict['Obesity']:
        shift_multiple(df, ['GBP', 'PGB'], +1)
        shift(df,'VPA', vpa_penalty)
        
    if input_dict['Bleeding disorders']:
        shift(df,'VPA', vpa_penalty)

    # Apply all the other rules, which were defined above
    for condition, changes in SIMPLE_RULES.items():
        if input_dict[condition]:
            for rule in changes:
                amount, drug_list = rule[0], rule[1]
                drug_list = [x for x in drug_list if x in list(df['drug'])]
                shift_multiple(df, drug_list, amount)
    
    """
    Hierarchy post-processing:
    Sometimes we have to massage the groups so they fit into Levels 1-4. 
    For example, if a drug in Group 1 is upgraded twice, it becomes Group -1. We need to fix this.
    """
    # However, this does not apply for drugs in locked groups. Those are not adjusted, so take them out.
    locked_drugs = df[df['locked']]
    unlocked_drugs = df[~df['locked']]

    groups = sorted(unlocked_drugs['group'].unique())
    group_mapping_dict = map_groups(groups)
    unlocked_drugs = unlocked_drugs.replace({"group": group_mapping_dict})
    df = pd.concat([unlocked_drugs, locked_drugs])

    """
    Final rule within the footnote of Table 1: 
    
    For focal seizures, if CLB comes in groups 1 or 2, move it to group 3; 
    For myoclonic and for unknown seizure types, if CLB comes in group 1, move it to group 2. 
    """
    if input_dict['Seizure type'] == 'Focal':
        row = df[df['drug'] == 'CLB']
        if list(row['group'])[0] < 3:
            df.loc[row, 'group'] = 3

    if (input_dict['Seizure type'] in ['Myoclonic', 'Primary generalized tonic-clonic + myoclonic', 
                                       "Primary generalized tonic-clonic + myoclonic + absence", "Myoclonic + absence", 
                                       "Unknown whether focal or generalized seizure type + age 21 or older", 
                                       "Unknown whether focal or generalized seizure type + age under 21"]):
        row = df[df['drug'] == 'CLB']
        if list(row['group'])[0] < 2:
            df.loc[row, 'group'] = 2
        
    # Build the output dictionary
    result = {level: df.loc[df['group'] == level, 'drug'].tolist() for level in sorted(df['group'].unique())}
    result = {int(k):v for k,v in result.items()}

    if top_only:
        best_level = min(level for level, drugs in result.items() if drugs)
        return {best_level: result[best_level]}

    return result

"""
Test cases in the EpiPick paper.
"""

"""
Test 1: a 35 year old man with focal seizures, who has diabetes and is taking non-contraceptive medications.
"""
input_data = {'Seizure type': 'Focal', 'Age': 35, 'Sex': 'Male', 'Diabetes mellitus': True, 'Co-medications (non-contraceptive)': True}
#epipick(input_data)

"""
Test 2: A 22 year old woman with primary generalized tonic-clonic + myoclonic + absence seizures, who is taking contraceptives and
has migraines.
"""
input_data = {'Seizure type': 'Primary generalized tonic-clonic + myoclonic + absence', 'Age': 22, 'Sex': 'Female', 'Oral contraceptives': True, 'Migraine': True}
#epipick(input_data)

"""
Test 3: a 67 year old man with focal seizures, who has a brain tumor and depression and is taking medications.
"""
input_data = {'Seizure type': 'Focal', 'Age': 67, 'Sex': 'Male', 'Brain tumor requiring chemo/radiation': True, 'Depression': True, 'Co-medications (non-contraceptive)': True}
print(epipick(input_data))