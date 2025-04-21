import spacy
from os.path import join 
import os
import json
import copy

import re

def contains_apostrophe(s):
    """
    Checks if a string contains any type of apostrophe.

    Parameters:
        s (str): The string to check.

    Returns:
        bool: True if the string contains any apostrophe; False otherwise.
    """
    # List of common apostrophe characters
    apostrophes = ["'", "‘", "’", "‛", "′", "`","'"]

    # Check if any of the apostrophe characters are in the string
    return any(char in s for char in apostrophes)

def clean_string(input_string):
    """
    Removes all characters from the input string except digits and letters (a-z, A-Z).

    Parameters:
        input_string (str): The string to process.

    Returns:
        str: The cleaned string containing only digits and letters.
    """
    # Use a regular expression to keep only alphanumeric characters
    return re.sub(r'[^a-zA-Z0-9]', '', input_string)


def spacy_merge(doc, task):
    out = []
    for w in doc:
        auto_merge_conditions = ["nt" == w.text]
        auto_sep_conditions = ["clock" in w.text, "them'll" in w.text, "girl's" in w.text, "L’Amour" in w.text, "y’" == w.text]
        if len(out) > 0:
            auto_merge_conditions += ["y’" == out[-1]["word"]]
            
            if task == "21styear" or task == "forgot":
                auto_merge_conditions += [("not" == w.text) and ("can" == out[-1]["word"])]
            if task == "black" or task == "forgot":
                auto_merge_conditions += [("na" == w.text) and ("gon" == out[-1]["word"])]

        if (contains_apostrophe(w.text) or any (auto_merge_conditions)) and not any(auto_sep_conditions):
            assert(len(out)>0)
            out[len(out)-1]["word"] = out[len(out)-1]["word"] + w.text
            out[len(out)-1]["pos"] = out[len(out)-1]["pos"] + "&" + w.pos_
        else:
            is_time = re.search(r'\d+:\d+', w.text) is not None
            if is_time:
                toks = w.text.split(":")
                out.append({"word":toks[0],"pos":w.pos_})
                out.append({"word":toks[1],"pos":""})
            elif w.text == "a.m.":
                out.append({"word":"a","pos":w.pos_})
                out.append({"word":"m","pos":""})

            else:
                out.append({"word":w.text,"pos":w.pos_})
    return out

def fix_cases(l):
    final = []
    
    for i in range(len(l)):
        if not l[i]["case"] == "success":
            if not len(final) == 0:
                final[-1]["pos"] += ("&"+l[i]["pos"])
        else:
            final.append(l[i])
    return final

def gen_pos_align(task, gentle_dir="/home/wsm32/project/wsm_thesis_scratch/narratives/stimuli/gentle/"):
    if spacy.prefer_gpu():
        print("Running on GPU")
    else:
        print("No GPU available, running on CPU")

    # Load the transformer model
    nlp = spacy.load("en_core_web_trf")

    with open(join(gentle_dir,task,"transcript.txt"), "r") as textfile:
        text = textfile.read()
        
    with open(join(gentle_dir,task,"align.json")) as alignfile:
        gentle = json.load(alignfile)['words']
    
    doc = nlp(text)
    merged = spacy_merge(doc, task)
    
    for w in merged:
        print(w["word"],w["pos"])
    out = []
    
    pos_idx = 0

    for align_idx in range(0,len(gentle)):
        print("Task:",task)
        print("Matching:",clean_string(gentle[align_idx]["word"]))
        count=0
        while True:
            print((merged[pos_idx]["word"]))
            if pos_idx >= len(merged):
                raise Exception("Could not align SPACY and GENTLE")
                return 0
            if clean_string(merged[pos_idx]["word"]) == clean_string(gentle[align_idx]["word"]):
                word_info = copy.deepcopy(gentle[align_idx])
                word_info["pos"] = merged[pos_idx]["pos"]
                out.append(word_info)
                pos_idx += 1
                break
            pos_idx += 1
            count += 1
            assert(count < 6)
    # If all that worked, now we need to fix the failed gentle cases
    out = fix_cases(out)


    outpath = join(gentle_dir,"pos",task,"pos_align.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath,"w") as out_file:
        json.dump(out, out_file, indent=4)
    return 1

for t in ["pieman","tunnel","lucy","prettymouth","milkywayoriginal","slumlordreach","notthefallintact","21styear","bronx","black","forgot"]:
#for t in ["21styear","bronx","black","forgot"]:
    gen_pos_align(t)

