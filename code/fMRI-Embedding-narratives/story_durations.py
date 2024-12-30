import json

# Tuples are (story start time, story end time)
data = {
    "pieman" : (15, 437),
    "tunnel" : (3, 1537),
    "lucy" : (3, 545),
    "prettymouth" : (21, 697),
    "milkywayoriginal": (21, 425),
    "slumlord" : (29.5, 932.5),
    "notthefallintact" : (29.5, 576.5),
    "21styear" : (21, 3599),
    "bronx" : (27, 27 + 536),
    "black" : (12, 12 + 800),
    "forgot" : (12, 12 + 837)
}

# Specify the output JSON file name
output_file = "duration.json"

# Write the dictionary to the JSON file
with open(output_file, "w") as json_file:
    json.dump(data, json_file, indent=4)  # indent=4 makes the JSON pretty-printed

print(f"Dictionary has been written to {output_file}")
