import bisect
import json
from os.path import join 
def extract_s(d):
    return d["start"]
def extract_e(d):
    return d["end"]
def get_pos_seq(task, tr_start, tr_end, tr_dur=1.5, buffer_dur=1.0, pos_dir="/home/wsm32/palmer_scratch/wsm_thesis_scratch/narratives/stimuli/gentle/pos", tr_offset=4.5):
    start_time = tr_start * tr_dur + buffer_dur + tr_offset
    end_time = tr_end * tr_dur - buffer_dur + tr_offset
    file_path = join(pos_dir,task,"pos_align.json")
    with open(file_path, "r") as in_file:
        align_data = json.load(in_file)
    left_corner_idx = bisect.bisect_left(align_data,start_time,key=extract_s)
    right_corner_idx = bisect.bisect_right(align_data,end_time,key=extract_e)
    seq = [e["pos"] for e in align_data[left_corner_idx:right_corner_idx]]
    joined = "&".join(seq)
    return joined.split("&")
