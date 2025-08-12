# Audio-Pattern-Recognition

Audio Pattern Recognition algorithms and study

## Project

Mental state prediction

### Parse dataset

```
python3 parse_android_metadata.py
```

### Read folds csv and convert to list

```
python3 convert_fold_list.py
```

### Extract features

```
python3 extract_features.py
```

Output:

```
Processed files: 225
Failed files: 0
```

### Merge the features in a table

```
python3 merge_all_features.py
```

Output

```
Found 225 feature files.
Files missing 'fold': 0 # nothing is missing
Frame count per fold:
1    1825619
2      80679
4      76353
5      71195
3      70637
Name: fold, dtype: int64
Unique files per fold:
fold
1    115
2     23
3     22
4     22
5     22
Name: file, dtype: int64
Total frames: 2124483
```

### Normalize features (training subset only)

Training subset is a fold 1.

```
python3 normalize_features.py
```

Output:

```
Unique files per fold:
fold
1    115
2     23
3     22
4     22
5     22
Name: file, dtype: int64

Frame count per fold:
1    1825619
2      80679
4      76353
5      71195
3      70637
Name: fold, dtype: int64

Sample files in fold 5:
['18_CM64_3.wav' '72_PM53_1.wav' '50_CF46_2.wav' '45_CF40_3.wav'
 '08_CF42_2.wav']
Saved normalized dataset.

```

### Train a baseline classifier

```
python3 train_baseline_classifier.py
```

Output:

```
Fold distribution:
1    1825619
2      80679
4      76353
5      71195
3      70637
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.7030
Frame-level F1 score: 0.6063
```

Seems fine but there's also class imbalance (only 9 files in fold 5 = test set), which might affect stability.

See the rest in workflow.log
