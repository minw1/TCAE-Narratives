import json

# Tuples are (story start time (tr-stim offset + silence/music), story end time, tr-stim offset)
# All numbers are in seconds (not TRs)
data = {
    "pieman" : (15, 437, 0),
    "tunnel" : (3, 1537, 3),
    "lucy" : (3, 545, 3),
    "prettymouth" : (21, 697, 0),
    "milkywayoriginal": (21, 425, 0),
    "slumlordreach" : (29.5, 932.5, 4.5),
    "notthefallintact" : (29.5, 576.5, 4.5),
    "21styear" : (21, 3359, 0),
    "bronx" : (27, 27 + 536, 12),
    "black" : (12, 12 + 800, 12),
    "forgot" : (12, 12 + 837, 12)
}

# Specify the output JSON file name
output_file = "duration.json"

# Write the dictionary to the JSON file
with open(output_file, "w") as json_file:
    json.dump(data, json_file, indent=4)  # indent=4 makes the JSON pretty-printed

print(f"Dictionary has been written to {output_file}")
