# NOTE: Please set the working directory to your main replication folder with os.chdir("/Users/main_replication_folder").
from __future__ import division
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.ensemble

import sklearn.linear_model
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
import sys
import os
#import package for updating python3
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

VERBOSE_FLAG = 1
pd.options.mode.chained_assignment = None # default = 'warn'
NUMJOBS = 2
NESTIMATORS = 1000
#the threshold must be numeric
np.set_printoptions(threshold=np.inf)


search_parameters = {}


def main(feature_set):

    infile = './data/feature_table.csv'

    justices = ['BREYER', 'GINSBURG', 'KENNEDY', 'ROBERTS', 'SCALIA']

    feature_names = ['amicus',
     'cutoffs_ALL', 'cutoffs_BREYER', 'cutoffs_GINSBURG', 'cutoffs_KENNEDY', 'cutoffs_ROBERTS', 'cutoffs_SCALIA',
    'BREYER_res_questions', 'GINSBURG_res_questions', 'KENNEDY_res_questions', 'ROBERTS_res_questions', 'SCALIA_res_questions',
    'BREYER_pet_questions', 'GINSBURG_pet_questions', 'KENNEDY_pet_questions', 'ROBERTS_pet_questions', 'SCALIA_pet_questions',
    'BREYER_question_diff', 'GINSBURG_question_diff', 'KENNEDY_question_diff', 'ROBERTS_question_diff', 'SCALIA_question_diff',
    'BREYER_cc_ratio_pet', 'GINSBURG_cc_ratio_pet', 'KENNEDY_cc_ratio_pet', 'ROBERTS_cc_ratio_pet', 'SCALIA_cc_ratio_pet',
    'BREYER_cc_ratio_res', 'GINSBURG_cc_ratio_res', 'KENNEDY_cc_ratio_res', 'ROBERTS_cc_ratio_res', 'SCALIA_cc_ratio_res',
    'BREYER_cc_ratio_diff', 'GINSBURG_cc_ratio_diff', 'KENNEDY_cc_ratio_diff', 'ROBERTS_cc_ratio_diff', 'SCALIA_cc_ratio_diff',
    'BREYER_qc_ratio_diff', 'GINSBURG_qc_ratio_diff', 'KENNEDY_qc_ratio_diff', 'ROBERTS_qc_ratio_diff', 'SCALIA_qc_ratio_diff',
    'BREYER_wc_ratio_diff', 'GINSBURG_wc_ratio_diff', 'KENNEDY_wc_ratio_diff', 'ROBERTS_wc_ratio_diff', 'SCALIA_wc_ratio_diff',
     'caseOrigin_circuit',
     'adminAction',
     'adminActionState',
     'lcDispositionDirection',
     'lcDisposition',
      'issueArea',
      'certReason', 'jurisdiction',
     ]

    feature_names_oa = ['amicus',
     'cutoffs_ALL', 'cutoffs_BREYER', 'cutoffs_GINSBURG', 'cutoffs_KENNEDY', 'cutoffs_ROBERTS', 'cutoffs_SCALIA',
    'BREYER_res_questions', 'GINSBURG_res_questions', 'KENNEDY_res_questions', 'ROBERTS_res_questions', 'SCALIA_res_questions',
    'BREYER_pet_questions', 'GINSBURG_pet_questions', 'KENNEDY_pet_questions', 'ROBERTS_pet_questions', 'SCALIA_pet_questions',
    'BREYER_question_diff', 'GINSBURG_question_diff', 'KENNEDY_question_diff', 'ROBERTS_question_diff', 'SCALIA_question_diff',
    'BREYER_cc_ratio_pet', 'GINSBURG_cc_ratio_pet', 'KENNEDY_cc_ratio_pet', 'ROBERTS_cc_ratio_pet', 'SCALIA_cc_ratio_pet',
    'BREYER_cc_ratio_res', 'GINSBURG_cc_ratio_res', 'KENNEDY_cc_ratio_res', 'ROBERTS_cc_ratio_res', 'SCALIA_cc_ratio_res',
    'BREYER_cc_ratio_diff', 'GINSBURG_cc_ratio_diff', 'KENNEDY_cc_ratio_diff', 'ROBERTS_cc_ratio_diff', 'SCALIA_cc_ratio_diff',
    'BREYER_qc_ratio_diff', 'GINSBURG_qc_ratio_diff', 'KENNEDY_qc_ratio_diff', 'ROBERTS_qc_ratio_diff', 'SCALIA_qc_ratio_diff',
    'BREYER_wc_ratio_diff', 'GINSBURG_wc_ratio_diff', 'KENNEDY_wc_ratio_diff', 'ROBERTS_wc_ratio_diff', 'SCALIA_wc_ratio_diff',
     ]

    feature_names_scdb = [
     'adminAction',
     'adminActionState',
     'lcDispositionDirection',
     'lcDisposition',
      'issueArea',
      'certReason', 'jurisdiction',
     ]

    if feature_set == 'scdb':
        feature_names = feature_names_scdb
    elif feature_set == 'oa':
        feature_names = feature_names_oa
    elif feature_set == 'all':
        feature_names = feature_names
    else:
        sys.exit("feature_names not recognized")


    d = pd.read_csv(infile, sep=',', index_col=0, encoding="ISO-8859-1")

    ## Break out decided cases
    decided_cases = d[pd.notnull(d.winner)]

    gs_results = sklearn.preprocessing.LabelEncoder().fit_transform(decided_cases.winner)

    for justice in justices:
        def map_diff(x):
            if x < -0.2:
                return -1
            elif x > 0.2:
                return 1
            else:
                return 0
        save_zero = lambda x : 1 if x == 0 else x
        div_ten_mod = lambda x :  x#10 if x > 100 else int(x/10)
        decided_cases[justice + '_pet_questions'] = (decided_cases[justice + '_pet_qc'] / decided_cases[justice + '_pet_cc'])
        decided_cases[justice + '_res_questions'] = (decided_cases[justice + '_res_qc'] / decided_cases[justice + '_res_cc'])
        decided_cases[justice + '_question_diff'] = (decided_cases[justice + '_pet_questions'] - decided_cases[justice + '_res_questions']).apply(map_diff)
        decided_cases[justice + '_avg_words_pet'] = (decided_cases[justice + '_pet_wc'] / decided_cases[justice + '_pet_cc']).apply(div_ten_mod)
        decided_cases[justice + '_avg_words_res'] = (decided_cases[justice + '_res_wc'] / decided_cases[justice + '_res_cc']).apply(div_ten_mod)
        decided_cases[justice + '_avg_words_diff'] = (decided_cases[justice + '_avg_words_pet'] - decided_cases[justice + '_avg_words_res']).apply(map_diff)
        decided_cases[justice + '_wc_ratio_pet'] = (decided_cases[justice + '_pet_wc'] / (decided_cases[justice  + '_pet_wc'] + decided_cases[justice  + '_res_wc']).apply(save_zero))
        decided_cases[justice + '_wc_ratio_res'] = (decided_cases[justice + '_res_wc'] / (decided_cases[justice  + '_pet_wc'] + decided_cases[justice  + '_res_wc']).apply(save_zero))
        decided_cases[justice + '_wc_ratio_diff'] = (decided_cases[justice + '_wc_ratio_pet'] - decided_cases[justice + '_wc_ratio_res']).apply(map_diff)
        decided_cases[justice + '_cc_ratio_pet'] = (decided_cases[justice + '_pet_cc'] / (decided_cases[justice  + '_pet_cc'] + decided_cases[justice  + '_res_cc']))
        decided_cases[justice + '_cc_ratio_res'] = (decided_cases[justice + '_res_cc'] / (decided_cases[justice  + '_pet_cc'] + decided_cases[justice  + '_res_cc']))
        decided_cases[justice + '_cc_ratio_diff'] = (decided_cases[justice + '_cc_ratio_pet'] - decided_cases[justice + '_cc_ratio_res']).apply(map_diff)
        decided_cases[justice + '_qc_ratio_pet'] = (decided_cases[justice + '_pet_qc'] / (decided_cases[justice  + '_pet_qc'] + decided_cases[justice  + '_res_qc']).apply(save_zero))
        decided_cases[justice + '_qc_ratio_res'] = (decided_cases[justice + '_res_qc'] / (decided_cases[justice  + '_pet_qc'] + decided_cases[justice  + '_res_qc']).apply(save_zero))
        decided_cases[justice + '_qc_ratio_diff'] = (decided_cases[justice + '_qc_ratio_pet'] - decided_cases[justice + '_qc_ratio_res']).apply(map_diff)

    map_test = lambda x : 0 if x != 13 else 1

    decided_cases['is_test_pet'] = decided_cases['petitioner_dk'].apply(map_test)
    decided_cases['is_test_res'] = decided_cases['respondent_dk'].apply(map_test)


    #print decided_cases

    gs_cases = decided_cases[feature_names].values.astype(np.float64)


    trees_model = sklearn.ensemble.ExtraTreesClassifier(n_estimators=1000,
                                                    class_weight="balanced_subsample",
                                                    warm_start=True,
                                                    n_jobs=-1)

    # Create the cross-validation fold
    # cv = sklearn.cross_validation.KFold(len(gs_cases), n_folds=10, shuffle=False)
    cv = KFold(n_splits=10, shuffle=False)
    # Create grid searcher
    # grid_search = sklearn.grid_search.GridSearchCV( trees_model, search_parameters,
    #      cv=cv, verbose=VERBOSE_FLAG,n_jobs=NUMJOBS)
    grid_search = GridSearchCV(AdaBoostClassifier(trees_model, n_estimators=NESTIMATORS), search_parameters,
                               cv=cv, verbose=VERBOSE_FLAG, n_jobs=NUMJOBS)

    # Fit model in grid search
    model = grid_search.fit(gs_cases, gs_results)

    print(model.cv_results_)
    print(model.best_estimator_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()

    for param_name in sorted(search_parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    best_search_parameters = {param_name : [best_parameters[param_name]] for param_name in search_parameters.keys()}

    print("Test accuracy score: %0.2f%%" % (model.best_score_ * 100))

    return model.best_score_


if __name__ == '__main__':
    #main("all")
    main("scdb")
    #main("oa")
