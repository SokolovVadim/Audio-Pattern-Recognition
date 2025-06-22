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

### For window_size = 20ms:

Fold distribution:
1    2738365
2     121013
4     114527
5     106789
3     105950
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.6929
Frame-level F1 score: 0.6038

### Window size = 100ms

window size = 100ms

At 16 kHz, that means:
win_length = 0.1 * 16000 = 1600 samples

But our n_fft is set to 1024, too small.
Librosa requires n_fft >= win_length

Change: n_fft = 2 ** int(np.ceil(np.log2(win_length)))  # Make n_fft the next power of 2

Fold distribution:
1    547729
2     24211
4     22914
5     21368
3     21199
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.7223
Frame-level F1 score: 0.6203

### Window size = 250ms, step_ms=50ms

Fold distribution:
1    547729
2     24211
4     22914
5     21368
3     21199
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.7725
Frame-level F1 score: 0.6757

### Window size = 500ms, step_ms=250ms

Fold distribution:
1    109599
2      4852
4      4592
5      4284
3      4247
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.7584
Frame-level F1 score: 0.6558

### Window size = 500ms, step_ms=100ms

Fold distribution:
1    273892
2     12110
4     11464
5     10688
3     10604
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.7833
Frame-level F1 score: 0.6860

### Window size = 1000ms, step_ms=200ms

Fold distribution:
1    136981
2      6062
4      5737
5      5351
3      5308
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.8015
Frame-level F1 score: 0.7087

### Window size = 5000ms, step_ms=1000ms

Fold distribution:
1    27449
2     1221
4     1158
5     1081
3     1070
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.8455
Frame-level F1 score: 0.7651


### Window size = 10000ms step_ms=2000ms

Fold distribution:
1    13758
2      617
4      584
5      546
3      540
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.8553
Frame-level F1 score: 0.7749

###  win_len_ms=5000, step_ms=500, iter=1000

Fold distribution:
1    54835
2     2431
4     2302
5     2148
3     2128
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.8515
Frame-level F1 score: 0.7733

### Window size = 5000ms, step_ms=2500ms

Fold distribution:
1    11019
2      492
4      469
5      439
3      434
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.8155
Frame-level F1 score: 0.7309

### Window size = 5000ms, step_ms=500ms, iter=2000

Fold distribution:
1    54835
2     2431
4     2302
5     2148
3     2128
Name: fold, dtype: int64
Files per fold:
1    72
4    16
3    13
5     9
2     6
Name: fold, dtype: int64
Frame-level accuracy: 0.8515
Frame-level F1 score: 0.7733

# -----------------------------------------------

### Window size = 20ms, step_ms=10ms, iter=1000

Fold distribution:
3    579138
4    570606
1    548142
5    524531
2    513203
Name: fold, dtype: int64
Files per fold:
5    236
2    221
3    195
4    175
1    150
Name: fold, dtype: int64
Train label distribution: 1    1182599
0    1028490
dtype: int64
Test label distribution: 0    308797
1    215734
dtype: int64

Frame-level accuracy: 0.6455
Frame-level F1 score: 0.6278

### Window size = 30ms, step_ms=15ms

Fold distribution:
3    386117
4    380433
1    365450
5    349728
2    342171

Name: fold, dtype: int64
Train label distribution: 1    788444
0    685727
dtype: int64
Test label distribution: 0    205887
1    143841
dtype: int64

Frame-level accuracy: 0.6506
Frame-level F1 score: 0.6296

### Window size = 100ms, step_ms=50ms

Fold distribution:
3    115910
4    114191
1    109688
5    104996
2    102732

Name: fold, dtype: int64
Train label distribution: 1    236633
0    205888
dtype: int64
Test label distribution: 0    61823
1    43173
dtype: int64

Frame-level accuracy: 0.6635
Frame-level F1 score: 0.6337

### Window size = 250ms, step_ms=125ms

Processed: 976 files
Failed: 1 files
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/57_CF25_3/57_CF25_3_2.wav: Audio too short (2880 samples) for win_length=4000

Fold distribution:
3    46435
4    45743
1    43925
5    42094
2    41205

Name: fold, dtype: int64
Train label distribution: 1    94754
0    82554
dtype: int64
Test label distribution: 0    24800
1    17294
dtype: int64

Frame-level accuracy: 0.6583
Frame-level F1 score: 0.6269

### Window size = 500ms, step_ms=250ms

Processed: 968 files
Failed: 9 files
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/48_CF34_3/48_CF34_3_4.wav: Audio too short (5680 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_7.wav: Audio too short (6481 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_8.wav: Audio too short (6865 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_32.wav: Audio too short (7633 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_13.wav: Audio too short (5345 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_22.wav: Audio too short (6865 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/54_CM48_2/54_CM48_2_5.wav: Audio too short (7840 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/54_CM48_2/54_CM48_2_13.wav: Audio too short (6209 samples) for win_length=8000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/57_CF25_3/57_CF25_3_2.wav: Audio too short (2880 samples) for win_length=8000

Fold distribution:
3    23311
4    22960
1    22031
5    21168
2    20736
Name: fold, dtype: int64
Files per fold:
5    234
2    215
3    195
4    175
1    149
Name: fold, dtype: int64
Train label distribution: 1    47512
0    41526
dtype: int64
Test label distribution: 0    12488
1     8680
dtype: int64

Frame-level accuracy: 0.6640
Frame-level F1 score: 0.6300

### Window size = 1000ms, step_ms=500ms

Filter out the outliers

Processed: 939 files
Failed: 38 files
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/05_CF41_3/05_CF41_3_9.wav: Audio too short (14608 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/05_CF41_3/05_CF41_3_4.wav: Audio too short (12160 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/06_CF44_2/06_CF44_2_7.wav: Audio too short (11777 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/07_CF50_2/07_CF50_2_4.wav: Audio too short (11008 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/12_CF36_1/12_CF36_1_11.wav: Audio too short (14033 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/12_CF36_1/12_CF36_1_7.wav: Audio too short (15776 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/12_CF36_1/12_CF36_1_1.wav: Audio too short (11040 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/13_CF45_2/13_CF45_2_23.wav: Audio too short (12800 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/23_PF55_2/23_PF55_2_7.wav: Audio too short (11440 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/31_CF55_2/31_CF55_2_5.wav: Audio too short (13232 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/31_CF55_2/31_CF55_2_8.wav: Audio too short (10720 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/31_CF55_2/31_CF55_2_14.wav: Audio too short (9056 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/32_CF22_3/32_CF22_3_8.wav: Audio too short (11248 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/33_PM19_3/33_PM19_3_5.wav: Audio too short (13504 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/33_PM19_3/33_PM19_3_3.wav: Audio too short (11569 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/46_PF44_2/46_PF44_2_5.wav: Audio too short (12000 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/48_CF34_3/48_CF34_3_16.wav: Audio too short (14816 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/48_CF34_3/48_CF34_3_4.wav: Audio too short (5680 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/50_CF46_2/50_CF46_2_15.wav: Audio too short (9648 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/50_CF46_2/50_CF46_2_10.wav: Audio too short (12976 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/50_CF46_2/50_CF46_2_14.wav: Audio too short (13168 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/51_CF52_2/51_CF52_2_8.wav: Audio too short (15281 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/52_CF29_3/52_CF29_3_8.wav: Audio too short (12241 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_7.wav: Audio too short (6481 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_8.wav: Audio too short (6865 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_32.wav: Audio too short (7633 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_13.wav: Audio too short (5345 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_22.wav: Audio too short (6865 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_12.wav: Audio too short (14880 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_20.wav: Audio too short (9168 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_14.wav: Audio too short (15264 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/53_CF52_1/53_CF52_1_6.wav: Audio too short (8400 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/54_CM48_2/54_CM48_2_5.wav: Audio too short (7840 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/54_CM48_2/54_CM48_2_13.wav: Audio too short (6209 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/56_CM23_3/56_CM23_3_6.wav: Audio too short (8368 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/57_CF25_3/57_CF25_3_2.wav: Audio too short (2880 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/70_PF51_1/70_PF51_1_7.wav: Audio too short (13233 samples) for win_length=16000
  - /home/vadim/ComputerScience/APR/Androids-Corpus/Interview-Task/audio_clip/70_PF51_1/70_PF51_1_6.wav: Audio too short (13632 samples) for win_length=16000

Fold distribution:
3    11771
4    11560
1    11116
5    10739
2    10526
Name: fold, dtype: int64
Files per fold:
5    226
2    205
3    191
4    170
1    147
Name: fold, dtype: int64
Train label distribution: 1    23923
0    21050
dtype: int64
Test label distribution: 0    6361
1    4378
dtype: int64

Frame-level accuracy: 0.6684
Frame-level F1 score: 0.6296

### Window size = 5000ms, step_ms=5000ms

Fold distribution:
3    1771
5    1766
4    1663
2    1577
1    1505
Name: fold, dtype: int64
Files per fold:
5    172
3    161
4    150
2    147
1    122
Name: fold, dtype: int64
Train label distribution: 0    3500
1    3016
dtype: int64
Test label distribution: 0    1134
1     632
dtype: int64

Frame-level accuracy: 0.7067
Frame-level F1 score: 0.5829

### Window size = 10000ms, step_ms=10000ms

Fold distribution:
5    1246
3    1227
4    1217
2    1083
1    1017
Name: fold, dtype: int64
Files per fold:
5    135
3    130
4    129
2    114
1    101
Name: fold, dtype: int64
Train label distribution: 0    2433
1    2111
dtype: int64
Test label distribution: 0    799
1    447
dtype: int64

Frame-level accuracy: 0.7255
Frame-level F1 score: 0.6400

### Window size = 15000ms, step_ms=15000ms

Processed: 487 files
Failed: 490 files

Fold distribution:
4    998
3    950
5    893
2    856
1    795
Name: fold, dtype: int64
Files per fold:
4    109
3    104
5     99
2     92
1     83
Name: fold, dtype: int64
Train label distribution: 0    1830
1    1769
dtype: int64
Test label distribution: 0    558
1    335
dtype: int64

Frame-level accuracy: 0.7290
Frame-level F1 score: 0.6849

### Window size = 20000ms, step_ms=20000ms

Fold distribution:
4    843
3    784
2    744
5    738
1    711
Name: fold, dtype: int64
Files per fold:
4    93
3    87
5    82
2    81
1    76
Name: fold, dtype: int64
Train label distribution: 1    1581
0    1501
dtype: int64
Test label distribution: 0    450
1    288
dtype: int64

Frame-level accuracy: 0.7317
Frame-level F1 score: 0.6786

### Window size = 30000ms, step_ms=30000ms

Processed: 327 files
Failed: 650 files

Fold distribution:
3    639
4    631
1    585
2    560
5    540
Name: fold, dtype: int64
Files per fold:
3    71
4    70
1    64
2    62
5    60
Name: fold, dtype: int64
Train label distribution: 1    1335
0    1080
dtype: int64
Test label distribution: 0    306
1    234
dtype: int64

Frame-level accuracy: 0.7074
Frame-level F1 score: 0.7074
