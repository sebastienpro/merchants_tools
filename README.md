# Retrieve the code 
Clone the code somewhere if this is the first time that you launch the scripts
```
git clone git@github.com:sebastienpro/merchants_tools.git
```
Retrieve the last version of the code if you ever have cloned the repo
```
git pull
```

# Install the dependencies
Just launch the following command
```
source init.sh
```

# Train the model
Put the json file which contain all the trainings brand in the current directory with the name `brand_classifier_data.json`

To train the model generate a new joblib file based on the brand_classifier_data.json datas launch the following command
```
./generate_brand_classifier_model.py brand_classifier_data.json
```
The generation will take some time but be patient ... and it will generate a joblib file with the version in his name exemple: `brand_classifier_v12.joblib`

# Test your new ML model
You can now test a joblib version with a specific data thanks to the script ``test_ml_with.py``
```
./test_ml_with.py brand_classifier_v12.joblib "Amazon.it*2V9NV3AE5"
```