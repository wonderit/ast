import csv
import json
from collections import OrderedDict

fieldnames = ("File_name","b1","File_path","b2", "Class_name", "b3", "Class_ID")

train = []
test = []
fold_number = 4
#the with statement is better since it handles closing your file properly after usage.
with open('3sec-data-audio-one-dir.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        entry = OrderedDict()
        file_path = row["File_path"]
        class_name = row["Class_name"]

        entry["wav"] = "./" + file_path.replace("3sec-", "")
        entry["labels"] = class_name.split(".")[1]
        #
        if i % 5 == fold_number:
            test.append(entry)
        else:
            train.append(entry)

output_train = {
    "data": train
}

output_test = {
    "data": test
}
with open(f'esc6_eval_data_{fold_number}.json', 'w') as jsonfile:
    json.dump(output_test, jsonfile)
    jsonfile.write('\n')

with open(f'esc6_train_data_{fold_number}.json', 'w') as jsonfile:
    json.dump(output_train, jsonfile)
    jsonfile.write('\n')