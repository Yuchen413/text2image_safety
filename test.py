import json

# Assuming you have a JSON file named 'data.json'
with open('data/vocab.json', 'r') as f:
    data = json.load(f)

# Now 'data' is a Python dictionary that contains the data from the JSON file
print(len(data))