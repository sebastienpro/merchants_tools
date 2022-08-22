#!/usr/bin/env python3
import os
import json
from joblib import load

brand_classifier = load(os.path.join("brand_classifier_2.joblib"))
f = open(os.path.join("brand_classifier_data.json"), "r")
brand_reference = json.load(f)
BRAND_CLASSIFIER_THRESHOLD = 0.9

def export_as_csv(ok, wrong_ok, fail):
    export_csv = open('result_test.csv', 'w')
    export_csv.writelines("Status;dirty name;returned clean name; expected name\n")
    for o in ok:
        export_csv.writelines(f'''OK;{o['dirty']};{o['clean']};{o['expected']}\n''')
    for w in wrong_ok:
        export_csv.writelines(f'''WRONG OK;{w['dirty']};{w['clean']};{w['expected']}\n''')
    for f in fail:
        export_csv.writelines(f'''FAIL;{f['dirty']};{f['clean']};{f['expected']}\n''')
    export_csv.close()

def main():
    fail = []
    wrong_ok = []
    ok = []
    nb_fail = 0
    nb_ok = 0
    nb_wrong_ok = 0

    source_test = json.load(open('sample_brand.json'))
    x = [merchant_values["dirty_name"] for merchant_values in source_test]
    score_matrix = brand_classifier.predict_proba(x)
    #print(x)
    for idx, prediction_scores in enumerate(score_matrix):
        (score, prediction) = max((v, i) for i, v in enumerate(prediction_scores))
        if score > 0.9:
            if brand_reference[prediction]['clean_name'] == source_test[idx]['expected_brand']:
                nb_ok += 1
                ok.append({
                    'dirty': source_test[idx]['dirty_name'],
                    'clean': brand_reference[prediction]['clean_name'],
                    'expected': source_test[idx]['expected_brand']}
                )
            else:
                nb_wrong_ok += 1
                wrong_ok.append({
                    'dirty': source_test[idx]['dirty_name'],
                    'clean': brand_reference[prediction]['clean_name'],
                    'expected': source_test[idx]['expected_brand']}
                )
                #print(f'''Returned: ({brand_reference[prediction]['clean_name']}) -> Expected: {source_test[idx]['expected_brand']}''')
            #print(str(brand_reference[prediction]['clean_name']) + " -> " + str(score))
        else:
            fail.append({
                'dirty': source_test[idx]['dirty_name'],
                'clean': brand_reference[prediction]['clean_name'],
                'expected': source_test[idx]['expected_brand']}
            )
            nb_fail += 1
            #print(f'''{source_test[idx]['dirty_name']} ({brand_reference[prediction]['clean_name']}) -> {source_test[idx]['expected_brand']} {score}''')
    print(f'''nb OK: {nb_ok}''')
    print(f'''nb WRONG OK: {nb_wrong_ok}''')
    print(f'''nb FAIL: {nb_fail}''')

    export_as_csv(ok, wrong_ok, fail)




if __name__ == "__main__":
    main()