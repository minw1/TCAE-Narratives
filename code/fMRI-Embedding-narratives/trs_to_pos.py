import bisect
import json
from os.path import join 
from utils import pos_tags

def extract_s(d):
    return d["start"]
def extract_e(d):
    return d["end"]
def get_pos_seq(align_data, tr_start, tr_end, stim_offset, tr_dur=1.5, buffer_dur=0.0, delay=4.5):
    start_time = tr_start * tr_dur + buffer_dur - delay - stim_offset
    end_time = tr_end * tr_dur - buffer_dur - delay - stim_offset
    left_corner_idx = bisect.bisect_left(align_data,start_time,key=extract_s)
    right_corner_idx = bisect.bisect_right(align_data,end_time,key=extract_e)
    #print(start_time, end_time)
    seq = [e["pos"] for e in align_data[left_corner_idx:right_corner_idx]]
    joined = "&".join(seq)
    split = joined.split("&")
    remove_empty = [i for i in split if i in pos_tags.keys()]
    #print(remove_empty)
    return [pos_tags[x] for x in remove_empty]
