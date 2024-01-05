import jsonlines
import os
import json
import pandas as pd

json_path = 'data/json'  # Make sure to use forward slash or double backslash for the path
output_jsonl_file = 'output.jsonl'

all_data = []

for json_file in os.listdir(json_path):
    if json_file.endswith(".json"):
        json_file_path = os.path.join(json_path, json_file)

        with open(json_file_path, 'r') as json_data:
            data = json.load(json_data)
            all_data.append(data)

            with jsonlines.open(output_jsonl_file, mode='a') as writer:
                writer.write(data)
