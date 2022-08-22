#!/usr/bin/env python3
import os
import json
from joblib import dump

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Classifier json file")
    return parser.parse_args()


def generate_model(source_file):
    f = open(source_file, "r")
    brand_reference = json.load(f)
    version = brand_reference["version"]
    print(version)

    x_train = []
    y_train = []
    count = 0
    for brand in brand_reference["brands"]:
        count_dirty = 0
        while count_dirty < 30:
            for dirty in brand.get("dirty_names")[:30]:
                count_dirty = count_dirty + 1
                x_train.append(dirty)
                y_train.append(count)
        count = count + 1

    text_clf = Pipeline(
        [
            ("vect", CountVectorizer(min_df=2)),
            ("clf", CalibratedClassifierCV(LinearSVC(max_iter=600), ensemble=False, cv=2)),
        ]
    )
    text_clf = text_clf.fit(x_train, y_train)

    dump(
        text_clf,
        os.path.join(os.path.dirname(os.path.realpath(__file__)), "brand_classifier_v" + str(version) + ".joblib"),
        compress=9,
    )


def main():
    args = get_args()
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file_name = args.json_file
    source_file = os.path.join(dir_path, file_name)
    generate_model(source_file)


if __name__ == "__main__":
    main()
