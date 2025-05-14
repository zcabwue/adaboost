# NOTE: Please set the working directory to your main replication folder with os.chdir("/Users/main_replication_folder").
import run_model_boost
import dup_model_mplus
import dup_model_courtcast
import dup_model_naive
import sys
import warnings
import os

if __name__ == '__main__':
    print("Running KKS model")
    boost_all = run_model_boost.main("all")
    boost_scdb = run_model_boost.main("scdb")
    boost_oa = run_model_boost.main("oa")

    print("Running M+ model")
    mplus_all = dup_model_mplus.main("all")
    mplus_scdb = dup_model_mplus.main("scdb")
    mplus_oa = dup_model_mplus.main("oa")

    print("Running CourtCast model")
    cc_all = dup_model_courtcast.main("all")
    cc_scdb = dup_model_courtcast.main("scdb")
    cc_oa = dup_model_courtcast.main("oa")

    print("Running Naive model")
    naive_all = dup_model_naive.main("all")
    naive_scdb = dup_model_naive.main("scdb")
    naive_oa = dup_model_naive.main("oa")

    print("KKS accuracy score on all data: %0.2f%%" % (boost_all * 100))
    print("KKS accuracy score on SCDB data: %0.2f%%" % (boost_scdb * 100))
    print("KKS accuracy score on OA data: %0.2f%%" % (boost_oa * 100))

    print("M+ accuracy score on all data: %0.2f%%" % (mplus_all * 100))
    print("M+ accuracy score on SCDB data: %0.2f%%" % (mplus_scdb * 100))
    print("M+ accuracy score on OA data: %0.2f%%" % (mplus_oa * 100))

    print("CourtCast accuracy score on all data: %0.2f%%" % (cc_all * 100))
    print("CourtCast accuracy score on SCDB data: %0.2f%%" % (cc_scdb * 100))
    print("CourtCast accuracy score on OA data: %0.2f%%" % (cc_oa * 100))

    print("Naive accuracy score on all data: %0.2f%%" % (naive_all * 100))
    print("Naive accuracy score on SCDB data: %0.2f%%" % (naive_scdb * 100))
    print("Naive accuracy score on OA data: %0.2f%%" % (naive_oa * 100))

