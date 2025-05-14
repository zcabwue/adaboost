# NOTE: Please set the working directory to your main replication folder with os.chdir("/Users/main_replication_folder").

# Imports
import copy
import numpy
import os
import pandas
import csv
import math
import sklearn.metrics
import time
import re
import numpy as np
from multiprocessing import freeze_support
from constants import party_map_data, court_circuit_map

def map_party(value):
    if value in party_map_data:
        return party_map_data[value]
    else:
        return None


def map_circuit(value):
    if value in court_circuit_map:
        return court_circuit_map[value]
    else:
        return None


if __name__ == '__main__':
    freeze_support()

    question_statement_dict = {}
    total_words_pet = {}
    total_words_res = {}
    justices = ['BREYER', 'GINSBURG', 'KENNEDY', 'ROBERTS', 'SCALIA']
    for justice in justices:
        question_statement_dict[justice] = {}
        docket = ""
        pet_questions = open('../data/questions/questions_' + justice + '_0.txt').read().split('\n')
        res_questions = open('../data/questions/questions_' + justice + '_1.txt').read().split('\n')
        for case in pet_questions:
            comments = list(filter(lambda x: not x.isspace() and not x=='', case.split('    ')))
            if len(comments) <= 2:
                comments = list(filter(lambda x: not x.isspace() and not x=='', case.split('  ')))
            num_words_pet = 0
            if comments:
                docket = re.search('\d+-\d+|\d+.* ', comments[0]).group(0).strip()
                comments = comments[1:]
                total_words_pet[docket] = 0
                question_statement_dict[justice][docket] = {}
                num_comments = len(comments)
                num_questions = 0
                for comment in comments:
                    num_words_pet += len(list((filter(lambda x : x not in ['-', '--'], comment.split()))))
                    total_words_pet[docket] += len(comment.split())
                    if '?' in comment:
                        num_questions += 1
                if num_comments > 0:
                    question_statement_dict[justice][docket][0] = (num_questions)
                else:
                    question_statement_dict[justice][docket][0] = 0
                question_statement_dict[justice][docket]['pet_wc'] = num_words_pet
                question_statement_dict[justice][docket]['pet_cc'] = 1 if num_comments == 0 else num_comments
        for case in res_questions:
            comments = list(filter(lambda x: not x.isspace() and not x=='', case.split('    ')))
            if len(comments) <= 2:
                comments = list(filter(lambda x: not x.isspace() and not x=='', case.split('  ')))
            num_words_res = 0
            if comments:
                docket = re.search('\d+-\d+|\d+.* ', comments[0]).group(0).strip()
                comments = comments[1:]
                total_words_res[docket] = 0
                if docket not in question_statement_dict[justice]:
                    question_statement_dict[justice][docket] = {}
                num_comments = len(comments)
                num_questions = 0
                for comment in comments:
                    num_words_res += len(list((filter(lambda x : x not in ['-', '--'], comment.split()))))
                    total_words_res[docket] += len(comment.split())
                    if '?' in comment:
                        num_questions += 1
                if num_comments > 0:
                    question_statement_dict[justice][docket][1] = (num_questions)
                else:
                    question_statement_dict[justice][docket][1] = 0
                question_statement_dict[justice][docket]['res_cc'] = 1 if num_comments == 0 else num_comments
            try:
                num_words_pet = question_statement_dict[justice][docket]['pet_wc']
            except KeyError:
                print(docket)
                num_words_pet = 0
            question_statement_dict[justice][docket]['pet_wc'] = (num_words_pet)#/float(num_words_pet + num_words_res)
            question_statement_dict[justice][docket]['res_wc'] = (num_words_res)#/float(num_words_pet + num_words_res)
    #print question_statement_dict

                    #print question



    # Load SCDB CSV data
    scdb_case_data = pandas.read_csv('../data/SCDB_2015_01_caseCentered_Citation_trimmed.csv', encoding='ISO-8859-1')
    scdb_case_data['petitioner_dk'] = scdb_case_data['petitioner'].apply(map_party)
    scdb_case_data['respondent_dk'] = scdb_case_data['respondent'].apply(map_party)
    scdb_case_data['caseOrigin_circuit'] = scdb_case_data['caseOrigin'].apply(map_circuit)
    scdb_case_data['caseSource_circuit'] = scdb_case_data['caseSource'].apply(map_circuit)
    scdb_case_data.loc[scdb_case_data['lcDisposition'] == 3, 'lcDisposition'] = 1.5
    scdb_case_data.loc[scdb_case_data['lcDispositionDirection'] == 3, 'lcDispositionDirection'] = 1.5
    scdb_case_data = scdb_case_data.set_index('docket').T.to_dict()

    with open('../data/feature_table_pre.txt', 'r') as csvinput:
        with open('../data/feature_table.csv', 'w',
                  newline='') as csvoutput:  # 'newline' to prevent double-spacing in Windows
            writer = csv.writer(csvoutput, delimiter=',', lineterminator='\n')
            reader = csv.reader(csvinput, delimiter='\t')

            all_rows = []  # Renamed from 'all' to 'all_rows' to avoid shadowing built-in function
            row = next(reader)
            additional_columns = ["term", 'petitioner_dk', 'respondent_dk', 'caseOrigin_circuit',
                                  'caseSource_circuit', 'source_origin_diff', 'lcDispositionDirection',
                                  'lcDisposition', 'issueArea', 'certReason', 'jurisdiction',
                                  'adminAction', 'adminActionState']
            for justice in justices:
                additional_columns += [justice + suffix for suffix in
                                       ('_pet_wc', '_res_wc', '_pet_qc', '_res_qc', '_pet_cc', '_res_cc')]

            row.extend(additional_columns)
            all_rows.append(row)

            for row in reader:
                docket = row[0]
                row[5] = row[5].replace(',', '')  # Changed to 'replace' for removing commas
                fields_to_try = ['term', 'petitioner_dk', 'respondent_dk', 'caseOrigin_circuit',
                                 'caseSource_circuit', 'lcDispositionDirection', 'lcDisposition',
                                 'issueArea', 'certReason', 'jurisdiction']

                for field in fields_to_try:
                    try:
                        row.append(scdb_case_data[docket][field])
                    except KeyError:
                        print(docket)
                        row.append(-1)

                try:
                    row.append(1 if scdb_case_data[docket]['caseSource_circuit'] == scdb_case_data[docket][
                        'caseOrigin_circuit'] else 0)
                except KeyError:
                    print(docket)
                    row.append(-1)

                for admin_field in ['adminAction', 'adminActionState']:
                    try:
                        value = scdb_case_data[docket][admin_field]
                        row.append(
                            1 if 0 < value <= (125 if admin_field == 'adminAction' else 63) and value != 124 else 0)
                    except KeyError:
                        print(docket)
                        row.append(-1)

                for justice in justices:
                    for suffix in ('_pet_wc', '_res_wc', '_pet_qc', '_res_qc', '_pet_cc', '_res_cc'):
                        try:
                            row.append(question_statement_dict[justice][docket][suffix.strip('_')])
                        except KeyError:
                            print(docket)
                            row.append(-1)

                row = list(map(lambda x: -1 if isinstance(x, float) and math.isnan(x) else x, row))
                all_rows.append(row)

            writer.writerows(all_rows)