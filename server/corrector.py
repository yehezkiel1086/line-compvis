# server/corrector.py
from collections import Counter

# ----------------------------------------------------------------
# helper: mapping / reconstruction (kept exactly as in logic)
# ----------------------------------------------------------------
MAPPING_D1 = {'0':['D','O','Q','U'],'1':['I','J','7'],'2':['S','Z','5'],'3':['B','E','8','9']}
MAPPING_D2 = {'0':['D','O','Q','U'],'1':['I','J'],'2':['Z'],'3':['E'],'4':['A','Y','H'],
              '5':['S'],'6':['G'],'7':['C'],'8':['B'],'9':['P','F']}
MAPPING_D3 = {'0':['D','O','Q','U'],'1':['I','J'],'2':['Z'],'3':['E'],'4':['Y'],
              '5':['S'],'6':['G'],'7':['C'],'8':[],'9':['P','F'],'A':['R'],'B':['H']}
MAPPING_D4 = {'0':['D','O','Q','U'],'1':['I','J'],'2':['Z'],'3':['E'],'4':['A','Y','H'],
              '5':['S'],'6':['G'],'7':['C'],'8':['B'],'9':['P','F']}
MAPPING_D5 = {'A':['R','4'],'B':['D','O','Q','8'],'C':['E','Z','2']}
MAPPING_D6 = {'1':['I','J'],'2':['Z'],'3':['E','B'],'4':['Y'],'5':['S'],'6':['G'],'7':['C']}
MAPPING_D7 = {'D':['B','H','O','Q','U','0','8']}

def map_by_position(c, pos):
    c = c.upper()
    mapping_sets = {
        0: MAPPING_D1, 1: MAPPING_D2, 2: MAPPING_D3, 3: MAPPING_D4,
        4: MAPPING_D5, 5: MAPPING_D6, 6: MAPPING_D7,
    }
    if pos in [7, 8, 9, 10]:
        mapping_sets[pos] = MAPPING_D2

    if pos in mapping_sets:
        for key, vals in mapping_sets[pos].items():
            if c == key or c in vals:
                return key
    return ""

def raw_digit_stats(list_raw):
    per_pos = [Counter() for _ in range(11)]
    for txt in list_raw:
        t = ''.join(k for k in txt.upper() if k.isalnum())[:11]
        for i, ch in enumerate(t):
            per_pos[i][ch] += 1
    return per_pos

def reconstruct_datecode(list_raw):
    if not list_raw:
        return ""

    raw_stats = raw_digit_stats(list_raw)
    per_pos = [[] for _ in range(11)]

    for txt in list_raw:
        t = ''.join(k for k in txt.upper() if k.isalnum())
        t = t[:11]
        for i in range(len(t)):
            mapped = map_by_position(t[i], i)
            if mapped:
                per_pos[i].append(mapped)

    result = ""
    for i in range(11):
        if per_pos[i]:
            result += Counter(per_pos[i]).most_common(1)[0][0]
        else:
            if i in [7, 8, 9, 10] and raw_stats[i]:
                result += raw_stats[i].most_common(1)[0][0]
            elif i == 0: result += '1'
            elif i == 6: result += 'D'
            elif i == 4: result += 'A'
            elif i == 5: result += '1'
            else:
                result += '0'
    return result

def stats_digit(list_raw):
    per_pos = [Counter() for _ in range(11)]
    for txt in list_raw:
        t = ''.join(k for k in txt.upper() if k.isalnum())[:11]
        for i, ch in enumerate(t):
            mapped = map_by_position(ch, i)
            if mapped:
                per_pos[i][mapped] += 1
    return per_pos

def majority_status(counter: Counter):
    if not counter: return "NO VALID"
    values = sorted(counter.values(), reverse=True)
    pattern = tuple(values)
    rules = {
        (5,): "VALID", (4,1): "VALID", (3,1,1): "VALID",
        (3,2): "WARNING", (2,2,1): "WARNING", (2,1,1,1): "WARNING",
        (1,1,1,1,1): "NO VALID", (4,): "VALID", (3,1): "VALID",
        (2,2): "WARNING", (2,1,1): "WARNING", (1,1,1,1): "NO VALID",
        (3,): "VALID", (2,1): "WARNING", (1,1,1): "NO VALID",
        (2,): "WARNING", (1,1): "NO VALID", (1,): "NO VALID",
    }
    return rules.get(pattern, "NO VALID")