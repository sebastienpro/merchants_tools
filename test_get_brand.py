#!/usr/bin/env python3
import argparse
import os
import json
from joblib import load

f = open(os.path.join("brand_classifier_data.json"), "r")
brand_reference = json.load(f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("ml_file", help="ML Classifier file")
    parser.add_argument("dirty_name", help="Dirty name to test")
    return parser.parse_args()


def main(ml_file, dirty_name):
    brand_classifier = load(os.path.join(ml_file))
    score_matrix = brand_classifier.predict_proba([dirty_name])
    for idx, prediction_scores in enumerate(score_matrix):
        (score, prediction) = max((v, i) for i, v in enumerate(prediction_scores))
        print("Clean Name -> %s"%brand_reference["brands"][prediction]["clean_name"])
        print("With Score: %f"%score)


if __name__ == "__main__":
    args = get_args()
    ml_file = args.ml_file
    dirty_name = args.dirty_name
    main(ml_file, dirty_name)