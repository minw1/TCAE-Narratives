import bisect
import json
from os.path import join 

pos_tags = {
    'ADJ': 0,
    'ADP': 1,
    'ADV': 2,
    'AUX': 3,
    'CCONJ': 4,
    'DET': 5,
    'INTJ': 6,
    'NOUN': 7,
    'NUM': 8,
    'PART': 9,
    'PRON': 10,
    'PROPN': 11,
    'PUNCT': 12,
    'SCONJ': 13,
    'SYM': 14,
    'VERB': 15,
    'X': 16,
    'PAD' : 100
}

def extract_s(d):
    return d["start"]
def extract_e(d):
    return d["end"]
def get_pos_seq(align_data, tr_start, tr_end, stim_offset, tr_dur=1.5, buffer_dur=0.0, delay=4.5):
    start_time = tr_start * tr_dur + buffer_dur - delay - stim_offset
    end_time = tr_end * tr_dur - buffer_dur - delay - stim_offset
    left_corner_idx = bisect.bisect_left(align_data,start_time,key=extract_s)
    right_corner_idx = bisect.bisect_right(align_data,end_time,key=extract_e)
    seq = [e["pos"] for e in align_data[left_corner_idx:right_corner_idx]]
    joined = "&".join(seq)
    return [pos_tags[x] for x in joined.split("&")]
