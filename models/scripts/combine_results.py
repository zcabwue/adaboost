# NOTE: Please set the working directory to your main replication folder with os.chdir("/Users/main_replication_folder").
import csv
import os

# All dicts keyed by docket.
case_kks_correct = {}
case_mplus_correct = {}
case_courtcast_correct = {}
case_decision_margin = {}
case_name = {}
case_term = {}

with open("results/results_kks.csv") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        docket = row["Docket"].strip()
        correct = 0
        if row["Predicted"] == row["Actual"]:
            correct = 1
        case_kks_correct[docket] = correct
        case_decision_margin[docket] = row["Majority Votes"]
        case_term[docket] = row["Term"]

with open("results/results_courtcast.csv") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        docket = row["docketnumber"]
        correct = 0
        if row["is_pred_correct"] == "True":
            correct = 1
        case_courtcast_correct[docket] = correct

with open("results/results_mplus.csv") as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        docket = row["docket"]
        case_name[docket] = row["caseName"]
        if (row["rf_correct_case"] != ""):
            correct = int(float(row["rf_correct_case"]))
            case_mplus_correct[docket] = correct


with open("results/results_combined.csv", 'w') as outfile:
    fieldnames = ["docket", "name", "term", "decision_margin", "kks_correct", "courtcast_correct", "mplus_correct"]
    writer = csv.DictWriter(outfile, fieldnames = fieldnames)
    writer.writeheader()
    for key in case_kks_correct:
        write_dict = {}
        write_dict["docket"] = key
        write_dict["term"] = case_term[key]
        write_dict["kks_correct"] = case_kks_correct[key]
        write_dict["decision_margin"] = case_decision_margin[key]
        if key in case_courtcast_correct:
            write_dict["courtcast_correct"] = case_courtcast_correct[key]
        else:
            write_dict["courtcast_correct"] = ""
        if key in case_mplus_correct:
            write_dict["mplus_correct"] = case_mplus_correct[key]
            write_dict["name"] = case_name[key]
        else:
            write_dict["mplus_correct"] = ""
            write_dict["name"] = ""
        writer.writerow(write_dict)

results_dict_kks = {}
results_dict_mplus = {}
results_dict_courtcast = {}
size_dict = {}

for key in case_decision_margin:
    try:
        result_kks = case_kks_correct[key]
        result_mplus = case_mplus_correct[key]
        results_courtcast = case_courtcast_correct[key]
    except KeyError:
        continue
    margin = case_decision_margin[key]
    if margin in size_dict:
        results_dict_kks[margin] += result_kks
        results_dict_mplus[margin] += result_mplus
        results_dict_courtcast[margin] += results_courtcast
        size_dict[margin] += 1.0
    else:
        results_dict_kks[margin] = result_kks
        results_dict_mplus[margin] = result_mplus
        results_dict_courtcast[margin] = results_courtcast
        size_dict[margin] = 1.0

for key in size_dict:
    results_dict_kks[key] = results_dict_kks[key] / size_dict[key]
    results_dict_mplus[key] = results_dict_mplus[key] / size_dict[key]
    results_dict_courtcast[key] = results_dict_courtcast[key] / size_dict[key]

print("KKS Results", results_dict_kks)
print("Mplus Results", results_dict_mplus)
print("Courtcast Results", results_dict_courtcast)