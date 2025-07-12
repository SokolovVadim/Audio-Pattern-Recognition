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

Recalculate after sorting out the doctor's voice.

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


----------------

### Add Confusion matrix

20ms

Frame-level accuracy: 0.6455
Frame-level F1 score: 0.6278
Confusion Matrix (in %):
[[34.66 24.21]
 [11.24 29.89]]

### Train Random Forest

### 20ms

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [25:10<00:00, 30.22s/it]

Frame-level accuracy: 0.6799
Frame-level F1 score: 0.6115
Confusion Matrix (in %):
[[42.8  16.07]
 [15.94 25.19]]

### 30ms

Fold distribution:
3    386117
4    380433
1    365450
5    349728
2    342171
Name: fold, dtype: int64
Files per fold:
5    236
2    221
3    195
4    175
1    150
Name: fold, dtype: int64
Train label distribution: 1    788444
0    685727
dtype: int64
Test label distribution: 0    205887
1    143841
dtype: int64
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [15:59<00:00, 19.19s/it]

Frame-level accuracy: 0.6829
Frame-level F1 score: 0.6115
Confusion Matrix (in %):
[[43.33 15.54]
 [16.17 24.96]]

### 100ms

Fold distribution:
3    115910
4    114191
1    109688
5    104996
2    102732
Name: fold, dtype: int64
Files per fold:
5    236
2    221
3    195
4    175
1    150
Name: fold, dtype: int64
Train label distribution: 1    236633
0    205888
dtype: int64
Test label distribution: 0    61823
1    43173
dtype: int64
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [03:43<00:00,  4.46s/it]

Frame-level accuracy: 0.6788
Frame-level F1 score: 0.6027
Confusion Matrix (in %):
[[43.51 15.37]
 [16.75 24.37]]

 ### 250


 Fold distribution:
3    46435
4    45743
1    43925
5    42094
2    41205
Name: fold, dtype: int64
Files per fold:
5    236
2    220
3    195
4    175
1    150
Name: fold, dtype: int64
Train label distribution: 1    94754
0    82554
dtype: int64
Test label distribution: 0    24800
1    17294
dtype: int64
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [01:10<00:00,  1.41s/it]

Frame-level accuracy: 0.6661
Frame-level F1 score: 0.5907
Confusion Matrix (in %):
[[42.52 16.4 ]
 [16.99 24.1 ]]

### 500

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:32<00:00,  1.55it/s]

Frame-level accuracy: 0.6618
Frame-level F1 score: 0.5945
Confusion Matrix (in %):
[[41.38 17.61]
 [16.21 24.79]]


### 1000

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:14<00:00,  3.47it/s]

Frame-level accuracy: 0.6589
Frame-level F1 score: 0.5957
Confusion Matrix (in %):
[[40.77 18.47]
 [15.64 25.12]]

### 5000

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 28.41it/s]

Frame-level accuracy: 0.6693
Frame-level F1 score: 0.5799
Confusion Matrix (in %):
[[44.11 20.1 ]
 [12.97 22.82]]

### 10000

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 38.58it/s]

Frame-level accuracy: 0.6894
Frame-level F1 score: 0.6006
Confusion Matrix (in %):
[[45.59 18.54]
 [12.52 23.35]]


### 15000

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 45.26it/s]

Frame-level accuracy: 0.6898
Frame-level F1 score: 0.6398
Confusion Matrix (in %):
[[41.43 21.05]
 [ 9.97 27.55]]


### 20000

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
100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 52.41it/s]

Frame-level accuracy: 0.6477
Frame-level F1 score: 0.5925
Confusion Matrix (in %):
[[39.16 21.82]
 [13.41 25.61]]


### SVM

### 30000ms

Frame-level accuracy: 0.7556
Frame-level F1 score: 0.7054
Confusion Matrix (in %):
[[46.3  10.37]
 [14.07 29.26]]



### 20000ms

Frame-level accuracy: 0.7575
Frame-level F1 score: 0.6427
Confusion Matrix (in %):
[[53.93  7.05]
 [17.21 21.82]]



### 15000ms

Frame-level accuracy: 0.7245
Frame-level F1 score: 0.6455
Confusion Matrix (in %):
[[47.37 15.12]
 [12.43 25.08]]

### 10000ms

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

Frame-level accuracy: 0.6758
Frame-level F1 score: 0.5765
Confusion Matrix (in %):
[[45.51 18.62]
 [13.8  22.07]]

### best parameters:
svc = SVC(C=0.3, gamma=0.01, class_weight={0: 1.0, 1: 0.8}, probability=True)

Frame-level accuracy: 0.7456
Frame-level F1 score: 0.6101
Confusion Matrix (in %):
[[54.65  9.47]
 [15.97 19.9 ]]


svc = SVC(C=0.3, gamma=0.005, class_weight={0: 1.0, 1: 0.8})

Frame-level accuracy: 0.7576
Frame-level F1 score: 0.6196
Confusion Matrix (in %):
[[56.02  8.11]
 [16.13 19.74]]

### 5000ms

Frame-level accuracy: 0.7186
Frame-level F1 score: 0.5217
Confusion Matrix (in %):
[[56.51  7.7 ]
 [20.44 15.35]]

### 1000ms

Frame-level accuracy: 0.6916
Frame-level F1 score: 0.5823
Confusion Matrix (in %):
[[47.66 11.57]
 [19.27 21.5 ]]

### 500ms

Frame-level accuracy: 0.6890
Frame-level F1 score: 0.5854
Confusion Matrix (in %):
[[46.94 12.05]
 [19.05 21.96]]

### 250ms

Frame-level accuracy: 0.6996
Frame-level F1 score: 0.6053
Confusion Matrix (in %):
[[46.93 11.98]
 [18.05 23.03]]

~50min

-----------

### feature importance analysis 

directly extracted and visualized feature importances.

Rms among the top features. That could indicate energy is a good mental state marker.

TODO: create a summary table 

frame size, top feature, 2nd, 3rd, accuracy, f1 score.

When we compute feature importances from a Random Forest using .feature_importances_, what we're actually getting is the Mean Decrease in Impurity (MDI). impurity refers to how mixed the class labels are in a node.
Higher MDI = Feature was used more often and split more samples while significantly reducing impurity thus, more important

### 30ms

Frame-level accuracy: 0.6913
Frame-level F1 score: 0.6303
Confusion Matrix (in %):
[[42.81 16.06]
 [14.81 26.32]]
      Feature  Importance
1      mfcc_2    0.080175
0      mfcc_1    0.079930
39        rms    0.051135
3      mfcc_4    0.032679
2      mfcc_3    0.030535
4      mfcc_5    0.029157
5      mfcc_6    0.027822
9     mfcc_10    0.026964
6      mfcc_7    0.026574
26   delta2_1    0.025626
7      mfcc_8    0.025196
12    mfcc_13    0.024370
8      mfcc_9    0.024081
11    mfcc_12    0.023530
10    mfcc_11    0.023302
27   delta2_2    0.023030
13    delta_1    0.022736
15    delta_3    0.021035
14    delta_2    0.020677
28   delta2_3    0.019789
18    delta_6    0.019469
30   delta2_5    0.019395
31   delta2_6    0.018944
17    delta_5    0.018786
19    delta_7    0.018772
29   delta2_4    0.018687
24   delta_12    0.018582
16    delta_4    0.018514
32   delta2_7    0.018317
20    delta_8    0.017796
25   delta_13    0.017775
23   delta_11    0.017667
21    delta_9    0.017648
22   delta_10    0.017484
33   delta2_8    0.017436
35  delta2_10    0.017433
37  delta2_12    0.017386
38  delta2_13    0.017222
34   delta2_9    0.017198
36  delta2_11    0.017142

### 100ms

Frame-level accuracy: 0.6871
Frame-level F1 score: 0.6238
Confusion Matrix (in %):
[[42.77 16.11]
 [15.18 25.94]]
      Feature  Importance
1      mfcc_2    0.081144
0      mfcc_1    0.078368
39        rms    0.053067
3      mfcc_4    0.030727
9     mfcc_10    0.030613
12    mfcc_13    0.028258
4      mfcc_5    0.028108
5      mfcc_6    0.027941
2      mfcc_3    0.027931
7      mfcc_8    0.026349
6      mfcc_7    0.025912
8      mfcc_9    0.024353
10    mfcc_11    0.024283
27   delta2_2    0.023914
11    mfcc_12    0.023690
26   delta2_1    0.021957
28   delta2_3    0.020659
31   delta2_6    0.020468
30   delta2_5    0.019538
37  delta2_12    0.019504
29   delta2_4    0.019322
32   delta2_7    0.019160
13    delta_1    0.018995
33   delta2_8    0.018765
38  delta2_13    0.018685
24   delta_12    0.018610
15    delta_3    0.018508
14    delta_2    0.018469
35  delta2_10    0.018323
36  delta2_11    0.018301
34   delta2_9    0.018281
18    delta_6    0.018189
17    delta_5    0.017772
16    delta_4    0.017621
20    delta_8    0.017570
23   delta_11    0.017489
19    delta_7    0.017488
22   delta_10    0.017284
25   delta_13    0.017195
21    delta_9    0.017187


### 250ms

Frame-level accuracy: 0.6743
Frame-level F1 score: 0.6108
Confusion Matrix (in %):
[[41.88 17.03]
 [15.53 25.55]]
      Feature  Importance
1      mfcc_2    0.080208
0      mfcc_1    0.079140
39        rms    0.056366
5      mfcc_6    0.031453
3      mfcc_4    0.031067
9     mfcc_10    0.030717
12    mfcc_13    0.030306
4      mfcc_5    0.029749
2      mfcc_3    0.027047
7      mfcc_8    0.025870
6      mfcc_7    0.025702
11    mfcc_12    0.024930
10    mfcc_11    0.024832
8      mfcc_9    0.024461
26   delta2_1    0.022921
27   delta2_2    0.021098
31   delta2_6    0.019795
28   delta2_3    0.019664
13    delta_1    0.019145
37  delta2_12    0.018960
38  delta2_13    0.018511
30   delta2_5    0.018492
29   delta2_4    0.018288
35  delta2_10    0.018177
14    delta_2    0.018169
32   delta2_7    0.018132
15    delta_3    0.018084
18    delta_6    0.018003
33   delta2_8    0.018003
36  delta2_11    0.017892
16    delta_4    0.017835
34   delta2_9    0.017668
22   delta_10    0.017592
24   delta_12    0.017572
23   delta_11    0.017509
17    delta_5    0.017472
20    delta_8    0.017335
19    delta_7    0.017333
21    delta_9    0.017277
25   delta_13    0.017224


### 500ms

Frame-level accuracy: 0.6670
Frame-level F1 score: 0.6097
Confusion Matrix (in %):
[[40.69 18.3 ]
 [14.99 26.01]]
      Feature  Importance
0      mfcc_1    0.081732
1      mfcc_2    0.077994
39        rms    0.056829
5      mfcc_6    0.035288
3      mfcc_4    0.031031
12    mfcc_13    0.030620
9     mfcc_10    0.030467
4      mfcc_5    0.028746
2      mfcc_3    0.025781
7      mfcc_8    0.025254
6      mfcc_7    0.025098
10    mfcc_11    0.025029
26   delta2_1    0.024768
11    mfcc_12    0.024251
8      mfcc_9    0.024024
13    delta_1    0.020616
27   delta2_2    0.019971
28   delta2_3    0.019582
31   delta2_6    0.019442
37  delta2_12    0.018585
30   delta2_5    0.018430
38  delta2_13    0.018409
29   delta2_4    0.018360
35  delta2_10    0.018161
15    delta_3    0.018117
32   delta2_7    0.018062
18    delta_6    0.017923
14    delta_2    0.017818
34   delta2_9    0.017813
36  delta2_11    0.017714
33   delta2_8    0.017645
23   delta_11    0.017638
22   delta_10    0.017570
24   delta_12    0.017452
16    delta_4    0.017429
19    delta_7    0.017386
25   delta_13    0.017336
17    delta_5    0.017269
20    delta_8    0.017258
21    delta_9    0.017104

### 1000ms

Frame-level accuracy: 0.6732
Frame-level F1 score: 0.6217
Confusion Matrix (in %):
[[40.46 18.77]
 [13.91 26.86]]
      Feature  Importance
0      mfcc_1    0.079808
1      mfcc_2    0.079372
39        rms    0.059258
5      mfcc_6    0.040267
3      mfcc_4    0.032422
12    mfcc_13    0.032405
9     mfcc_10    0.030521
4      mfcc_5    0.027769
11    mfcc_12    0.025543
2      mfcc_3    0.025253
6      mfcc_7    0.024455
10    mfcc_11    0.024412
7      mfcc_8    0.024219
8      mfcc_9    0.024214
26   delta2_1    0.023927
13    delta_1    0.020768
27   delta2_2    0.020513
28   delta2_3    0.019114
31   delta2_6    0.018518
15    delta_3    0.018430
37  delta2_12    0.018203
38  delta2_13    0.018077
34   delta2_9    0.017946
32   delta2_7    0.017936
14    delta_2    0.017689
30   delta2_5    0.017684
33   delta2_8    0.017671
24   delta_12    0.017641
29   delta2_4    0.017602
35  delta2_10    0.017528
23   delta_11    0.017401
18    delta_6    0.017358
16    delta_4    0.017273
22   delta_10    0.017259
36  delta2_11    0.017134
20    delta_8    0.017073
17    delta_5    0.017005
25   delta_13    0.016879
19    delta_7    0.016822
21    delta_9    0.016629


### 5000ms

Frame-level accuracy: 0.6999
Frame-level F1 score: 0.6159
Confusion Matrix (in %):
[[45.92 18.29]
 [11.72 24.07]]
      Feature  Importance
1      mfcc_2    0.088538
0      mfcc_1    0.069074
39        rms    0.059860
12    mfcc_13    0.038557
3      mfcc_4    0.028634
5      mfcc_6    0.027133
30   delta2_5    0.026299
27   delta2_2    0.025885
14    delta_2    0.024610
26   delta2_1    0.023239
37  delta2_12    0.023040
9     mfcc_10    0.022634
28   delta2_3    0.022591
17    delta_5    0.022517
31   delta2_6    0.022022
13    delta_1    0.020964
11    mfcc_12    0.020956
29   delta2_4    0.020914
33   delta2_8    0.020842
22   delta_10    0.020755
15    delta_3    0.020554
21    delta_9    0.020414
18    delta_6    0.020369
24   delta_12    0.020052
16    delta_4    0.019965
36  delta2_11    0.019395
23   delta_11    0.019367
34   delta2_9    0.019340
25   delta_13    0.019287
2      mfcc_3    0.019067
20    delta_8    0.018888
38  delta2_13    0.018886
35  delta2_10    0.018811
19    delta_7    0.018612
8      mfcc_9    0.017852
32   delta2_7    0.017599
6      mfcc_7    0.016579
4      mfcc_5    0.016039
7      mfcc_8    0.015913
10    mfcc_11    0.013948


### 10000ms

Frame-level accuracy: 0.7600
Frame-level F1 score: 0.6940
Confusion Matrix (in %):
[[48.8  15.33]
 [ 8.67 27.21]]
      Feature  Importance
1      mfcc_2    0.091659
0      mfcc_1    0.059903
39        rms    0.053524
12    mfcc_13    0.035271
26   delta2_1    0.030656
14    delta_2    0.029393
30   delta2_5    0.026545
34   delta2_9    0.025691
35  delta2_10    0.025626
22   delta_10    0.025501
21    delta_9    0.025417
27   delta2_2    0.024513
31   delta2_6    0.024300
24   delta_12    0.023139
32   delta2_7    0.022791
18    delta_6    0.022746
5      mfcc_6    0.022717
16    delta_4    0.021847
3      mfcc_4    0.021730
28   delta2_3    0.021697
33   delta2_8    0.021563
13    delta_1    0.020897
38  delta2_13    0.020754
29   delta2_4    0.020625
23   delta_11    0.019940
11    mfcc_12    0.019935
15    delta_3    0.019615
17    delta_5    0.019542
19    delta_7    0.018581
25   delta_13    0.018526
9     mfcc_10    0.018515
36  delta2_11    0.018174
20    delta_8    0.017911
8      mfcc_9    0.017435
2      mfcc_3    0.016612
37  delta2_12    0.016065
10    mfcc_11    0.015886
4      mfcc_5    0.015471
6      mfcc_7    0.015250
7      mfcc_8    0.014037

### 15000ms

Frame-level accuracy: 0.6831
Frame-level F1 score: 0.6440
Confusion Matrix (in %):
[[39.64 22.84]
 [ 8.85 28.67]]
      Feature  Importance
0      mfcc_1    0.087504
1      mfcc_2    0.082794
39        rms    0.054640
5      mfcc_6    0.035003
3      mfcc_4    0.031623
21    delta_9    0.030899
14    delta_2    0.029198
12    mfcc_13    0.029108
29   delta2_4    0.024647
19    delta_7    0.024518
16    delta_4    0.023534
17    delta_5    0.023130
15    delta_3    0.022494
20    delta_8    0.022417
9     mfcc_10    0.022379
28   delta2_3    0.022325
27   delta2_2    0.022172
34   delta2_9    0.021869
33   delta2_8    0.021271
26   delta2_1    0.021203
31   delta2_6    0.020850
35  delta2_10    0.020598
30   delta2_5    0.020205
24   delta_12    0.020008
13    delta_1    0.019201
23   delta_11    0.018801
37  delta2_12    0.018721
7      mfcc_8    0.018695
36  delta2_11    0.018387
32   delta2_7    0.018322
38  delta2_13    0.017536
2      mfcc_3    0.017096
25   delta_13    0.016967
22   delta_10    0.016645
10    mfcc_11    0.014975
4      mfcc_5    0.014911
18    delta_6    0.014728
8      mfcc_9    0.014686
6      mfcc_7    0.013655
11    mfcc_12    0.012284

### 20000ms

Frame-level accuracy: 0.7398
Frame-level F1 score: 0.6883
Confusion Matrix (in %):
[[45.26 15.72]
 [10.3  28.73]]
      Feature  Importance
1      mfcc_2    0.067511
0      mfcc_1    0.066745
12    mfcc_13    0.061329
39        rms    0.060580
14    delta_2    0.031572
3      mfcc_4    0.030088
5      mfcc_6    0.030082
33   delta2_8    0.025890
26   delta2_1    0.023813
16    delta_4    0.023679
25   delta_13    0.023637
29   delta2_4    0.023222
27   delta2_2    0.022870
21    delta_9    0.022803
35  delta2_10    0.022464
15    delta_3    0.022363
19    delta_7    0.022208
30   delta2_5    0.021512
38  delta2_13    0.021381
9     mfcc_10    0.021189
20    delta_8    0.020199
22   delta_10    0.019968
7      mfcc_8    0.019631
2      mfcc_3    0.019041
28   delta2_3    0.018987
17    delta_5    0.018984
24   delta_12    0.018835
31   delta2_6    0.018685
36  delta2_11    0.018506
11    mfcc_12    0.018259
32   delta2_7    0.018009
6      mfcc_7    0.017382
8      mfcc_9    0.016965
10    mfcc_11    0.016523
34   delta2_9    0.016400
18    delta_6    0.016324
23   delta_11    0.016199
13    delta_1    0.015987
4      mfcc_5    0.015198
37  delta2_12    0.014981


### 30000ms

Frame-level accuracy: 0.7741
Frame-level F1 score: 0.7645
Confusion Matrix (in %):
[[40.74 15.93]
 [ 6.67 36.67]]
      Feature  Importance
1      mfcc_2    0.085605
0      mfcc_1    0.071842
5      mfcc_6    0.053549
39        rms    0.046374
12    mfcc_13    0.039878
3      mfcc_4    0.031389
28   delta2_3    0.030553
34   delta2_9    0.026070
11    mfcc_12    0.025910
26   delta2_1    0.025606
6      mfcc_7    0.022481
15    delta_3    0.022266
21    delta_9    0.022092
35  delta2_10    0.021208
27   delta2_2    0.021137
9     mfcc_10    0.020845
25   delta_13    0.020551
14    delta_2    0.020509
17    delta_5    0.020468
31   delta2_6    0.020104
29   delta2_4    0.019934
20    delta_8    0.019751
16    delta_4    0.019479
7      mfcc_8    0.019463
18    delta_6    0.019026
8      mfcc_9    0.018984
38  delta2_13    0.018842
4      mfcc_5    0.018579
32   delta2_7    0.018265
30   delta2_5    0.018124
13    delta_1    0.017394
2      mfcc_3    0.016921
10    mfcc_11    0.016759
23   delta_11    0.016678
33   delta2_8    0.016537
22   delta_10    0.016342
19    delta_7    0.016128
24   delta_12    0.015993
37  delta2_12    0.014488
36  delta2_11    0.013877


------------------

## Added features

Linear

### 20ms

Frame-level accuracy: 0.6441
Frame-level F1 score: 0.6270
Confusion Matrix (in %):
[[34.5  24.37]
 [11.22 29.91]]

### 30ms

Fold distribution:
3    386117
4    380433
1    365450
5    349728
2    342171
Name: fold, dtype: int64
Files per fold:
5    236
2    221
3    195
4    175
1    150
Name: fold, dtype: int64
Train label distribution: 1    788444
0    685727
dtype: int64
Test label distribution: 0    205887
1    143841
dtype: int64

Frame-level accuracy: 0.6491
Frame-level F1 score: 0.6297
Confusion Matrix (in %):
[[35.08 23.79]
 [11.3  29.83]]


### 100ms

Frame-level accuracy: 0.6648
Frame-level F1 score: 0.6376
Confusion Matrix (in %):
[[36.99 21.89]
 [11.63 29.49]]

### 250ms

Frame-level accuracy: 0.6602
Frame-level F1 score: 0.6311
Confusion Matrix (in %):
[[36.95 21.94]
 [12.04 29.06]]

### 500ms

Frame-level accuracy: 0.6663
Frame-level F1 score: 0.6350
Confusion Matrix (in %):
[[37.6  21.32]
 [12.05 29.03]]

### 1000ms

Frame-level accuracy: 0.6809
Frame-level F1 score: 0.6468
Confusion Matrix (in %):
[[38.88 20.09]
 [11.82 29.22]]

### 5000ms

Frame-level accuracy: 0.7201
Frame-level F1 score: 0.6813
Confusion Matrix (in %):
[[42.1  16.42]
 [11.57 29.92]]

### 10000ms

Frame-level accuracy: 0.7225
Frame-level F1 score: 0.6793
Confusion Matrix (in %):
[[42.87 15.03]
 [12.72 29.38]]

### 15000ms

Frame-level accuracy: 0.7110
Frame-level F1 score: 0.6951
Confusion Matrix (in %):
[[38.15 19.36]
 [ 9.54 32.95]]

### 20000ms

Frame-level accuracy: 0.6680
Frame-level F1 score: 0.6529
Confusion Matrix (in %):
[[35.57 21.34]
 [11.86 31.23]]

### 30000ms

Frame-level accuracy: 0.7000
Frame-level F1 score: 0.6939
Confusion Matrix (in %):
[[36.   17.33]
 [12.67 34.  ]]

-----------------

### Random Forest

### 20ms

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
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [49:04<00:00, 29.45s/it]

Frame-level accuracy: 0.6820
Frame-level F1 score: 0.6290

File-level Accuracy (majority vote): 0.7966
File-level F1 Score: 0.6923
Confusion Matrix (in %):
[[56.78 13.98]
 [ 6.36 22.88]]


### 30ms

Frame-level accuracy: 0.6849
Frame-level F1 score: 0.6281

File-level Accuracy (majority vote): 0.7966
File-level F1 Score: 0.6883
Confusion Matrix (in %):
[[57.2  13.56]
 [ 6.78 22.46]]


### 100ms

Frame-level accuracy: 0.6813
Frame-level F1 score: 0.6224

File-level Accuracy (majority vote): 0.7839
File-level F1 Score: 0.6752
Confusion Matrix (in %):
[[55.93 14.83]
 [ 6.78 22.46]]

### 250ms

Frame-level accuracy: 0.6703
Frame-level F1 score: 0.6130

File-level Accuracy (majority vote): 0.7585
File-level F1 Score: 0.6460
Confusion Matrix (in %):
[[53.81 16.95]
 [ 7.2  22.03]]

### 500ms

Frame-level accuracy: 0.6684
Frame-level F1 score: 0.6144

File-level Accuracy (majority vote): 0.7393
File-level F1 Score: 0.6303
Confusion Matrix (in %):
[[51.71 18.8 ]
 [ 7.26 22.22]]

### 1000ms

Frame-level accuracy: 0.6715
Frame-level F1 score: 0.6272

File-level Accuracy (majority vote): 0.7168
File-level F1 Score: 0.6279
Confusion Matrix (in %):
[[47.79 22.57]
 [ 5.75 23.89]]

### 5000ms

Frame-level accuracy: 0.7437
Frame-level F1 score: 0.7087

File-level Accuracy (majority vote): 0.8023
File-level F1 Score: 0.7302
Confusion Matrix (in %):
[[53.49 13.95]
 [ 5.81 26.74]]

### 10000ms

Frame-level accuracy: 0.7495
Frame-level F1 score: 0.7162

File-level Accuracy (majority vote): 0.8148
File-level F1 Score: 0.7475
Confusion Matrix (in %):
[[54.07 11.11]
 [ 7.41 27.41]]


### 15000ms

Frame-level accuracy: 0.7399
Frame-level F1 score: 0.7239

File-level Accuracy (majority vote): 0.7576
File-level F1 Score: 0.7000
Confusion Matrix (in %):
[[47.47 15.15]
 [ 9.09 28.28]]

### 20000ms

Frame-level accuracy: 0.6957
Frame-level F1 score: 0.6883

File-level Accuracy (majority vote): 0.7317
File-level F1 Score: 0.7027
Confusion Matrix (in %):
[[41.46 19.51]
 [ 7.32 31.71]]


### 30000ms

Frame-level accuracy: 0.7733
Frame-level F1 score: 0.7792

File-level Accuracy (majority vote): 0.8000
File-level F1 Score: 0.7931
Confusion Matrix (in %):
[[41.67 15.  ]
 [ 5.   38.33]]

-----------------

### SVM

#### 20ms

#### 30ms



#### 100ms

Frame-level accuracy: 0.7169
Frame-level F1 score: 0.6154

File-level Accuracy (majority vote): 0.8263
File-level F1 Score: 0.6963
Confusion Matrix (in %):
[[62.71  8.05]
 [ 9.32 19.92]]

#### 250ms

Frame-level accuracy: 0.7044
Frame-level F1 score: 0.5912

File-level Accuracy (majority vote): 0.8263
File-level F1 Score: 0.6822
Confusion Matrix (in %):
[[63.98  6.78]
 [10.59 18.64]]

#### 500ms

Frame-level accuracy: 0.6893
Frame-level F1 score: 0.5662

File-level Accuracy (majority vote): 0.7949
File-level F1 Score: 0.6190
Confusion Matrix (in %):
[[62.82  7.69]
 [12.82 16.67]]

#### 1000ms

Frame-level accuracy: 0.6903
Frame-level F1 score: 0.5646

File-level Accuracy (majority vote): 0.7876
File-level F1 Score: 0.6000
Confusion Matrix (in %):
[[62.83  7.52]
 [13.72 15.93]]

#### 5000ms

Frame-level accuracy: 0.7465
Frame-level F1 score: 0.6573

File-level Accuracy (majority vote): 0.8140
File-level F1 Score: 0.6735
Confusion Matrix (in %):
[[62.21  5.23]
 [13.37 19.19]]

#### 10000ms

Train label distribution: 1    2406
0    2027
dtype: int64
Test label distribution: 0    601
1    437
dtype: int64
Train: [2027 2406]
Test: [601 437]

Frame-level accuracy: 0.7495
Frame-level F1 score: 0.6790

File-level Accuracy (majority vote): 0.8000
File-level F1 Score: 0.6897
Confusion Matrix (in %):
[[57.78  7.41]
 [12.59 22.22]]

### 15000ms

Frame-level accuracy: 0.7428
Frame-level F1 score: 0.6942

File-level Accuracy (majority vote): 0.7273
File-level F1 Score: 0.6494
Confusion Matrix (in %):
[[47.47 15.15]
 [12.12 25.25]]


### 20000ms

Frame-level accuracy: 0.7589
Frame-level F1 score: 0.6806

File-level Accuracy (majority vote): 0.7317
File-level F1 Score: 0.6071
Confusion Matrix (in %):
[[52.44  8.54]
 [18.29 20.73]]

### 30000ms

Frame-level accuracy: 0.7333
Frame-level F1 score: 0.7059

File-level Accuracy (majority vote): 0.7333
File-level F1 Score: 0.7037
Confusion Matrix (in %):
[[41.67 15.  ]
 [11.67 31.67]]
