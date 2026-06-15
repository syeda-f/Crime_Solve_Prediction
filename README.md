# Crime-Solve-Prediction
Built a decision tree classifier to predict whether a crime will be solved using 638,454 US crime incidents (1980-2014). The model identifies key predictors of crime clearance and quantifies temporal drift over 35 years.

## Key Results
- **84.7% accuracy** after fixing data leakage (down from 99.97% fake accuracy)
- **30% reduction in false positives** through hyperparameter tuning
- **Victim-perpetrator relationship** identified as #1 predictor
- **5.9% model drift** measured over 35 years using temporal split

## Tool Used
- WEKA (J48 implementation of C4.5 decision tree)

## What I Learned
- Data leakage from perpetrator attributes caused 99.97% fake accuracy
- Aggressive pruning (C=0.01, M=10000) produces simpler, more generalizable trees
- Crime solving patterns changed significantly around 1998-2000
- Models trained on historical data need retraining every 5-10 years

## Train final model
weka.classifiers.trees.J48 -C 0.01 -M 10000 -t crime_train.arff -x 10

## Test on validation set
weka.classifiers.trees.J48 -C 0.01 -M 10000 -t crime_train.arff -T crime_val.arff

## Test on test set
weka.classifiers.trees.J48 -C 0.01 -M 10000 -t crime_train.arff -T crime_test.arff
