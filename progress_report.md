
4/30 640pm
                    name        ic  elapsed_time  ...  colsample_bytree  min_child_weight  reg_lambda
5            max_depth_3  0.199125      0.449065  ...               0.8                10         1.0
9     learning_rate_0.03  0.198875      0.779673  ...               0.8                10         1.0
23   min_child_weight_15  0.186442      0.290555  ...               0.8                15         1.0
10    learning_rate_0.04  0.185380      0.284556  ...               0.8                10         1.0
14         subsample_0.7  0.182428      0.630921  ...               0.8                10         1.0
18  colsample_bytree_0.7  0.179345      0.436613  ...               0.7                10         1.0
24   min_child_weight_20  0.178424      0.298541  ...               0.8                20         1.0
21    min_child_weight_5  0.178049      0.270887  ...               0.8                 5         1.0
27        reg_lambda_2.0  0.178039      0.258051  ...               0.8                10         2.0
1       n_estimators_200  0.177826      0.346578  ...               0.8                10         1.0
26        reg_lambda_0.8  0.177826      0.246033  ...               0.8                10         0.8
25        reg_lambda_0.5  0.177826      0.236651  ...               0.8                10         0.5
0               baseline  0.177826      0.497971  ...               0.8                10         1.0
4       n_estimators_600  0.177826      0.359577  ...               0.8                10         1.0
3       n_estimators_500  0.177826      0.325851  ...               0.8                10         1.0
2       n_estimators_300  0.177826      0.325536  ...               0.8                10         1.0
12    learning_rate_0.07  0.177825      0.313054  ...               0.8                10         1.0
11    learning_rate_0.06  0.177825      0.263538  ...               0.8                10         1.0
28        reg_lambda_5.0  0.177759      0.578109  ...               0.8                10         5.0
22    min_child_weight_7  0.177679      0.271035  ...               0.8                 7         1.0
19  colsample_bytree_0.9  0.175868      0.310542  ...               0.9                10         1.0
15         subsample_0.9  0.174487      0.378052  ...               0.8                10         1.0
7            max_depth_6  0.169704      0.278538  ...               0.8                10         1.0
8            max_depth_7  0.166989      0.314540  ...               0.8                10         1.0
20  colsample_bytree_1.0  0.166551      0.328059  ...               1.0                10         1.0
6            max_depth_4  0.163878      0.391064  ...               0.8                10         1.0
16         subsample_1.0  0.159993      0.225549  ...               0.8                10         1.0
13         subsample_0.6  0.153751      0.319540  ...               0.8                10         1.0
17  colsample_bytree_0.6  0.134267      0.252542  ...               0.6                10         1.0

depth = 3 !! 
why depth 6 fail why did 0.03 lr help?
interpret^

explain features
explain model
show experiments
explain outcomes

=== Best values by parameter group ===

=== n_estimators ranking ===
            name       ic  n_estimators
n_estimators_200 0.177826           200
        baseline 0.177826           400
n_estimators_600 0.177826           600
n_estimators_500 0.177826           500
n_estimators_300 0.177826           300


=== max_depth ranking ===
       name       ic  max_depth
max_depth_3 0.199125          3
   baseline 0.177826          5
max_depth_6 0.169704          6
max_depth_7 0.166989          7
max_depth_4 0.163878          4

=== learning_rate ranking ===
              name       ic  learning_rate
learning_rate_0.03 0.198875           0.03
learning_rate_0.04 0.185380           0.04
          baseline 0.177826           0.05
learning_rate_0.07 0.177825           0.07
learning_rate_0.06 0.177825           0.06

=== subsample ranking ===
         name       ic  subsample
subsample_0.7 0.182428        0.7
     baseline 0.177826        0.8
subsample_0.9 0.174487        0.9
subsample_1.0 0.159993        1.0
subsample_0.6 0.153751        0.6

=== colsample_bytree ranking ===
                name       ic  colsample_bytree
colsample_bytree_0.7 0.179345               0.7
            baseline 0.177826               0.8
colsample_bytree_0.9 0.175868               0.9
colsample_bytree_1.0 0.166551               1.0
colsample_bytree_0.6 0.134267               0.6

=== min_child_weight ranking ===
               name       ic  min_child_weight
min_child_weight_15 0.186442                15
min_child_weight_20 0.178424                20
 min_child_weight_5 0.178049                 5
           baseline 0.177826                10
 min_child_weight_7 0.177679                 7

=== reg_lambda ranking ===
          name       ic  reg_lambda
reg_lambda_2.0 0.178039         2.0
reg_lambda_0.8 0.177826         0.8
reg_lambda_0.5 0.177826         0.5
      baseline 0.177826         1.0
reg_lambda_5.0 0.177759         5.0

---------
2nd run

=== Final Results ===
                     name        ic  elapsed_time
5             max_depth_3  0.199125      0.614421
14     learning_rate_0.03  0.198875      1.038925
9     learning_rate_0.005  0.196080      0.971397
10     learning_rate_0.01  0.194235      0.659890
12     learning_rate_0.02  0.189204      0.657630
33    min_child_weight_25  0.186786      0.395530
11    learning_rate_0.015  0.186720      0.503812
30    min_child_weight_17  0.186453      0.426720
29    min_child_weight_15  0.186442      0.359131
16     learning_rate_0.04  0.185380      0.451159
19          subsample_0.7  0.182428      0.854934
13    learning_rate_0.025  0.181129      0.518050
20         subsample_0.75  0.179390      0.496999
23  colsample_bytree_0.65  0.179345      0.595994
24   colsample_bytree_0.7  0.179345      0.571925
32    min_child_weight_20  0.178424      0.357074
27     min_child_weight_5  0.178049      0.369524
35         reg_lambda_2.0  0.178039      0.366740
34         reg_lambda_1.5  0.177850      0.375815
26   colsample_bytree_0.8  0.177826      0.400279
0                baseline  0.177826      1.600270
21          subsample_0.8  0.177826      0.370026
2        n_estimators_300  0.177826      0.444771
3        n_estimators_500  0.177826      0.726660
4        n_estimators_600  0.177826      0.568743
1        n_estimators_200  0.177826      0.427447
28    min_child_weight_12  0.177818      0.416692
31    min_child_weight_18  0.177800      0.385247
41         reg_lambda_5.0  0.177759      0.471768
40         reg_lambda_4.5  0.177759      0.371525
38         reg_lambda_3.5  0.177666      0.374162
39         reg_lambda_4.0  0.177664      0.426520
25  colsample_bytree_0.75  0.177172      0.412608
36         reg_lambda_2.5  0.176963      0.413724
37         reg_lambda_3.0  0.176963      0.364228
18         subsample_0.65  0.174428      0.491089
15    learning_rate_0.035  0.171196      0.667276
7             max_depth_6  0.169704      0.477079
8             max_depth_7  0.166989      0.456180
6             max_depth_4  0.163878      0.593189
17          subsample_0.6  0.153751      0.508630
22   colsample_bytree_0.6  0.134267      0.368303

=== Best values by parameter group ===

=== n_estimators ranking ===
            name       ic  n_estimators
        baseline 0.177826           400
n_estimators_300 0.177826           300
n_estimators_500 0.177826           500
n_estimators_600 0.177826           600
n_estimators_200 0.177826           200

=== max_depth ranking ===
       name       ic  max_depth
max_depth_3 0.199125          3
   baseline 0.177826          5
max_depth_6 0.169704          6
max_depth_7 0.166989          7
max_depth_4 0.163878          4

=== learning_rate ranking ===
               name       ic  learning_rate
 learning_rate_0.03 0.198875          0.030
learning_rate_0.005 0.196080          0.005
 learning_rate_0.01 0.194235          0.010
 learning_rate_0.02 0.189204          0.020
learning_rate_0.015 0.186720          0.015
 learning_rate_0.04 0.185380          0.040
learning_rate_0.025 0.181129          0.025
           baseline 0.177826          0.050
learning_rate_0.035 0.171196          0.035

=== subsample ranking ===
          name       ic  subsample
 subsample_0.7 0.182428       0.70
subsample_0.75 0.179390       0.75
      baseline 0.177826       0.80
 subsample_0.8 0.177826       0.80
subsample_0.65 0.174428       0.65
 subsample_0.6 0.153751       0.60

=== colsample_bytree ranking ===
                 name       ic  colsample_bytree
colsample_bytree_0.65 0.179345              0.65
 colsample_bytree_0.7 0.179345              0.70
 colsample_bytree_0.8 0.177826              0.80
             baseline 0.177826              0.80
colsample_bytree_0.75 0.177172              0.75
 colsample_bytree_0.6 0.134267              0.60

=== min_child_weight ranking ===
               name       ic  min_child_weight
min_child_weight_25 0.186786                25
min_child_weight_17 0.186453                17
min_child_weight_15 0.186442                15
min_child_weight_20 0.178424                20
 min_child_weight_5 0.178049                 5
           baseline 0.177826                10
min_child_weight_12 0.177818                12
min_child_weight_18 0.177800                18

=== reg_lambda ranking ===
          name       ic  reg_lambda
reg_lambda_2.0 0.178039         2.0
reg_lambda_1.5 0.177850         1.5
      baseline 0.177826         1.0
reg_lambda_5.0 0.177759         5.0
reg_lambda_4.5 0.177759         4.5
reg_lambda_3.5 0.177666         3.5
reg_lambda_4.0 0.177664         4.0
reg_lambda_2.5 0.176963         2.5
reg_lambda_3.0 0.176963         3.0

---------4

d, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  .apply(_per_stock_features)
>> Creating 3 splits
    split 1: train=27,229, val=4,960
    split 2: train=71,393, val=4,974
    split 3: train=115,641, val=4,972

>> Running 36 XGBoost configs experiments

---baseline---
IC avg: 0.1208 | std: 0.0493 | splits: [0.0576, 0.1269, 0.1778]

---max_depth_3---
IC avg: 0.1162 | std: 0.0718 | splits: [0.024, 0.1254, 0.1991]

---max_depth_4---
IC avg: 0.1086 | std: 0.0518 | splits: [0.0394, 0.1226, 0.1639]

---max_depth_6---
IC avg: 0.1107 | std: 0.0579 | splits: [0.032, 0.1304, 0.1697]

---max_depth_7---
IC avg: 0.0974 | std: 0.0611 | splits: [0.0183, 0.1068, 0.167]

---learning_rate_0.005---
IC avg: 0.1283 | std: 0.0566 | splits: [0.0576, 0.1311, 0.1961]

---learning_rate_0.01---
IC avg: 0.1255 | std: 0.0558 | splits: [0.0576, 0.1245, 0.1942]

---learning_rate_0.015---
IC avg: 0.1234 | std: 0.0527 | splits: [0.0576, 0.1259, 0.1867]

---learning_rate_0.02---
IC avg: 0.1257 | std: 0.0538 | splits: [0.0576, 0.1302, 0.1892]

---learning_rate_0.025---
IC avg: 0.1230 | std: 0.0507 | splits: [0.0576, 0.1303, 0.1811]

---learning_rate_0.03---
IC avg: 0.1308 | std: 0.0578 | splits: [0.0576, 0.136, 0.1989]

---learning_rate_0.035---
IC avg: 0.1195 | std: 0.0469 | splits: [0.0576, 0.1298, 0.1712]

---learning_rate_0.04---
IC avg: 0.1250 | std: 0.0524 | splits: [0.0576, 0.1321, 0.1854]

---subsample_0.6---
IC avg: 0.1095 | std: 0.0440 | splits: [0.0494, 0.1253, 0.1538]

---subsample_0.65---
IC avg: 0.1110 | std: 0.0549 | splits: [0.0404, 0.1181, 0.1744]

---subsample_0.7---
IC avg: 0.1237 | std: 0.0463 | splits: [0.0693, 0.1192, 0.1824]

---subsample_0.75---
IC avg: 0.1285 | std: 0.0411 | splits: [0.0788, 0.1273, 0.1794]

---subsample_0.8---
IC avg: 0.1208 | std: 0.0493 | splits: [0.0576, 0.1269, 0.1778]

---colsample_bytree_0.6---
IC avg: 0.1004 | std: 0.0419 | splits: [0.0414, 0.1257, 0.1343]

---colsample_bytree_0.65---
IC avg: 0.1260 | std: 0.0417 | splits: [0.0776, 0.1209, 0.1793]

---colsample_bytree_0.7---
IC avg: 0.1260 | std: 0.0417 | splits: [0.0776, 0.1209, 0.1793]

---colsample_bytree_0.75---
IC avg: 0.1153 | std: 0.0489 | splits: [0.0576, 0.111, 0.1772]

---colsample_bytree_0.8---
IC avg: 0.1208 | std: 0.0493 | splits: [0.0576, 0.1269, 0.1778]

---min_child_weight_10---
IC avg: 0.1208 | std: 0.0493 | splits: [0.0576, 0.1269, 0.1778]

---min_child_weight_20---
IC avg: 0.1231 | std: 0.0496 | splits: [0.058, 0.1328, 0.1784]

---min_child_weight_25---
IC avg: 0.1239 | std: 0.0530 | splits: [0.057, 0.1278, 0.1868]

---min_child_weight_27---
IC avg: 0.1230 | std: 0.0528 | splits: [0.057, 0.1256, 0.1862]

---min_child_weight_30---
IC avg: 0.1234 | std: 0.0527 | splits: [0.058, 0.1251, 0.1871]

---reg_lambda_1.5---
IC avg: 0.1237 | std: 0.0498 | splits: [0.0576, 0.1356, 0.1779]

---reg_lambda_2.0---
IC avg: 0.1211 | std: 0.0494 | splits: [0.0576, 0.1276, 0.178]

---reg_lambda_2.5---
IC avg: 0.1228 | std: 0.0494 | splits: [0.0576, 0.1339, 0.177]

---reg_lambda_3.0---
IC avg: 0.1221 | std: 0.0492 | splits: [0.0576, 0.1317, 0.177]

---reg_lambda_3.5---
IC avg: 0.1234 | std: 0.0497 | splits: [0.0576, 0.1348, 0.1777]

---reg_lambda_4.0---
IC avg: 0.1197 | std: 0.0491 | splits: [0.0576, 0.1238, 0.1777]

---reg_lambda_4.5---
IC avg: 0.1201 | std: 0.0490 | splits: [0.0581, 0.1246, 0.1778]

---reg_lambda_5.0---
IC avg: 0.1218 | std: 0.0492 | splits: [0.0581, 0.1295, 0.1778]

=== Final Results ===
                 name       ic   ic_std  split1_ic  split2_ic  split3_ic  elapsed_time  n_estimators  max_depth  learning_rate  subsample  colsample_bytree  min_child_weight  reg_lambda
   learning_rate_0.03 0.130825 0.057792   0.057598   0.136001   0.198875      1.370033           400          5          0.030       0.80              0.80                10         1.0
       subsample_0.75 0.128509 0.041071   0.078807   0.127331   0.179390      0.902301           400          5          0.050       0.75              0.80                10         1.0
  learning_rate_0.005 0.128259 0.056571   0.057598   0.131097   0.196080      1.961384           400          5          0.005       0.80              0.80                10         1.0
colsample_bytree_0.65 0.125953 0.041682   0.077624   0.120889   0.179345      0.953799           400          5          0.050       0.80              0.65                10         1.0
 colsample_bytree_0.7 0.125953 0.041682   0.077624   0.120889   0.179345      0.985965           400          5          0.050       0.80              0.70                10         1.0
   learning_rate_0.02 0.125668 0.053823   0.057598   0.130200   0.189204      0.920302           400          5          0.020       0.80              0.80                10         1.0
   learning_rate_0.01 0.125454 0.055785   0.057598   0.124528   0.194235      1.208377           400          5          0.010       0.80              0.80                10         1.0
   learning_rate_0.04 0.125026 0.052406   0.057598   0.132099   0.185380      0.703614           400          5          0.040       0.80              0.80                10         1.0
  min_child_weight_25 0.123874 0.053043   0.057036   0.127799   0.186786      0.811848           400          5          0.050       0.80              0.80                25         1.0
       reg_lambda_1.5 0.123689 0.049812   0.057598   0.135619   0.177850      0.824838           400          5          0.050       0.80              0.80                10         1.5
        subsample_0.7 0.123652 0.046282   0.069324   0.119205   0.182428      0.935166           400          5          0.050       0.70              0.80                10         1.0
  min_child_weight_30 0.123419 0.052715   0.058045   0.125075   0.187138      0.762072           400          5          0.050       0.80              0.80                30         1.0
  learning_rate_0.015 0.123411 0.052743   0.057598   0.125914   0.186720      0.985729           400          5          0.015       0.80              0.80                10         1.0
       reg_lambda_3.5 0.123365 0.049683   0.057598   0.134832   0.177666      1.177694           400          5          0.050       0.80              0.80                10         3.5
  min_child_weight_20 0.123099 0.049624   0.058045   0.132830   0.178424      0.824410           400          5          0.050       0.80              0.80                20         1.0
  learning_rate_0.025 0.123015 0.050695   0.057598   0.130317   0.181129      0.834687           400          5          0.025       0.80              0.80                10         1.0
  min_child_weight_27 0.122960 0.052764   0.057036   0.125647   0.186198      0.778582           400          5          0.050       0.80              0.80                27         1.0
       reg_lambda_2.5 0.122813 0.049354   0.057598   0.133878   0.176963      0.814886           400          5          0.050       0.80              0.80                10         2.5
       reg_lambda_3.0 0.122071 0.049199   0.057598   0.131652   0.176963      0.812442           400          5          0.050       0.80              0.80                10         3.0
       reg_lambda_5.0 0.121768 0.049169   0.058063   0.129481   0.177759      0.835752           400          5          0.050       0.80              0.80                10         5.0
       reg_lambda_2.0 0.121077 0.049385   0.057598   0.127594   0.178039      0.888712           400          5          0.050       0.80              0.80                10         2.0
  min_child_weight_10 0.120774 0.049273   0.057598   0.126898   0.177826      0.755138           400          5          0.050       0.80              0.80                10         1.0
             baseline 0.120774 0.049273   0.057598   0.126898   0.177826      0.887704           400          5          0.050       0.80              0.80                10         1.0
 colsample_bytree_0.8 0.120774 0.049273   0.057598   0.126898   0.177826      0.785783           400          5          0.050       0.80              0.80                10         1.0
        subsample_0.8 0.120774 0.049273   0.057598   0.126898   0.177826      0.763861           400          5          0.050       0.80              0.80                10         1.0
       reg_lambda_4.5 0.120134 0.048967   0.058063   0.124581   0.177759      0.934558           400          5          0.050       0.80              0.80                10         4.5
       reg_lambda_4.0 0.119696 0.049103   0.057598   0.123825   0.177664      0.920430           400          5          0.050       0.80              0.80                10         4.0
  learning_rate_0.035 0.119534 0.046941   0.057598   0.129808   0.171196      0.805638           400          5          0.035       0.80              0.80                10         1.0
          max_depth_3 0.116178 0.071810   0.023962   0.125446   0.199125      0.672426           400          3          0.050       0.80              0.80                10         1.0
colsample_bytree_0.75 0.115257 0.048908   0.057598   0.111001   0.177172      0.758829           400          5          0.050       0.80              0.75                10         1.0
       subsample_0.65 0.110992 0.054933   0.040436   0.118112   0.174428      0.690146           400          5          0.050       0.65              0.80                10         1.0
          max_depth_6 0.110704 0.057935   0.031970   0.130437   0.169704      0.775650           400          6          0.050       0.80              0.80                10         1.0
        subsample_0.6 0.109487 0.044027   0.049432   0.125278   0.153751      0.779646           400          5          0.050       0.60              0.80                10         1.0
          max_depth_4 0.108612 0.051768   0.039397   0.122561   0.163878      0.704767           400          4          0.050       0.80              0.80                10         1.0
 colsample_bytree_0.6 0.100438 0.041924   0.041355   0.125691   0.134267      0.956576           400          5          0.050       0.80              0.60                10         1.0
          max_depth_7 0.097351 0.061060   0.018312   0.106752   0.166989      0.746588           400          7          0.050       0.80              0.80                10         1.0

=== Final Results ===
                 name       ic   ic_std  split1_ic  split2_ic  split3_ic  elapsed_time
 colsample_bytree_0.7 0.135809 0.043362   0.088805   0.125205   0.193419      0.802007
colsample_bytree_0.65 0.135809 0.043362   0.088805   0.125205   0.193419      0.798054
   learning_rate_0.02 0.124220 0.058763   0.047397   0.135185   0.190078      1.538879
   learning_rate_0.01 0.124201 0.060081   0.047397   0.131133   0.194073      1.698061
  learning_rate_0.015 0.123838 0.059658   0.047397   0.131136   0.192982      1.695575
        subsample_0.7 0.120668 0.056822   0.050989   0.120840   0.190174      1.096971
       subsample_0.65 0.120645 0.059498   0.044898   0.126787   0.190249      1.285627
  learning_rate_0.005 0.120194 0.055869   0.047397   0.129992   0.183192      2.462216
  learning_rate_0.035 0.119278 0.056123   0.047397   0.126071   0.184365      1.027613
 colsample_bytree_0.6 0.118897 0.058150   0.047522   0.119209   0.189959      1.153488
   learning_rate_0.03 0.118416 0.055547   0.047397   0.124848   0.183002      1.246418
        subsample_0.6 0.118066 0.070647   0.028689   0.124084   0.201425      1.531506
  learning_rate_0.025 0.117934 0.053540   0.047397   0.129365   0.177039      1.352996
       reg_lambda_3.5 0.116978 0.056140   0.047397   0.118657   0.184881      0.851671
       subsample_0.75 0.116719 0.050683   0.045391   0.146267   0.158498      1.079013
          max_depth_3 0.116439 0.074462   0.023425   0.120189   0.205703      0.823867
          max_depth_4 0.116340 0.058765   0.043467   0.118175   0.187377      1.036157
       reg_lambda_5.0 0.114812 0.051499   0.048168   0.122696   0.173574      0.874478
       reg_lambda_4.0 0.114401 0.051957   0.047607   0.121280   0.174316      1.030507
       reg_lambda_4.5 0.113934 0.051634   0.048168   0.119337   0.174299      0.942126
   learning_rate_0.04 0.113827 0.051946   0.047397   0.119878   0.174207      1.208778
       reg_lambda_3.0 0.113195 0.052327   0.047397   0.116767   0.175422      0.837222
       reg_lambda_1.5 0.111062 0.047102   0.047397   0.125924   0.159865      0.795417
       reg_lambda_2.5 0.110878 0.049534   0.047397   0.116966   0.168270      0.863730
       reg_lambda_2.0 0.110775 0.047013   0.047397   0.125065   0.159863      0.749904
        subsample_0.8 0.110256 0.046702   0.047397   0.124133   0.159239      0.856910
             baseline 0.110256 0.046702   0.047397   0.124133   0.159239      2.202271
  min_child_weight_10 0.110256 0.046702   0.047397   0.124133   0.159239      0.745911
 colsample_bytree_0.8 0.110256 0.046702   0.047397   0.124133   0.159239      0.772312
  min_child_weight_27 0.109473 0.045084   0.048628   0.123393   0.156397      0.716006
  min_child_weight_25 0.109463 0.045081   0.048628   0.123365   0.156397      0.899025
colsample_bytree_0.75 0.109168 0.045983   0.047397   0.122449   0.157657      0.720806
  min_child_weight_20 0.109106 0.044698   0.048628   0.123420   0.155271      0.881362
  min_child_weight_30 0.108748 0.044817   0.048628   0.121430   0.156186      0.712688
          max_depth_6 0.095460 0.043184   0.035841   0.113806   0.136734      1.192619
          max_depth_7 0.058735 0.039001   0.012104   0.107561   0.056541      1.179673


=== Final Results ===
                 name       ic   ic_std  split1_ic  split2_ic  split3_ic  elapsed_time
 colsample_bytree_0.7 0.135809 0.043362   0.088805   0.125205   0.193419      0.997767
colsample_bytree_0.65 0.135809 0.043362   0.088805   0.125205   0.193419      0.965718
   learning_rate_0.02 0.124220 0.058763   0.047397   0.135185   0.190078      1.528674
   learning_rate_0.01 0.124201 0.060081   0.047397   0.131133   0.194073      2.043385
  learning_rate_0.015 0.123838 0.059658   0.047397   0.131136   0.192982      1.848551
        subsample_0.7 0.120668 0.056822   0.050989   0.120840   0.190174      1.340499
       subsample_0.65 0.120645 0.059498   0.044898   0.126787   0.190249      1.564616
  learning_rate_0.005 0.120194 0.055869   0.047397   0.129992   0.183192      3.520183
  learning_rate_0.035 0.119278 0.056123   0.047397   0.126071   0.184365      1.528565
 colsample_bytree_0.6 0.118897 0.058150   0.047522   0.119209   0.189959      1.186366
   learning_rate_0.03 0.118416 0.055547   0.047397   0.124848   0.183002      1.791529
        subsample_0.6 0.118066 0.070647   0.028689   0.124084   0.201425      1.703658
  learning_rate_0.025 0.117934 0.053540   0.047397   0.129365   0.177039      1.803344
       reg_lambda_3.5 0.116978 0.056140   0.047397   0.118657   0.184881      1.080565
       subsample_0.75 0.116719 0.050683   0.045391   0.146267   0.158498      1.011723
          max_depth_3 0.116439 0.074462   0.023425   0.120189   0.205703      1.646759
          max_depth_4 0.116340 0.058765   0.043467   0.118175   0.187377      1.175500
       reg_lambda_5.0 0.114812 0.051499   0.048168   0.122696   0.173574      0.940373
       reg_lambda_4.0 0.114401 0.051957   0.047607   0.121280   0.174316      1.161457
       reg_lambda_4.5 0.113934 0.051634   0.048168   0.119337   0.174299      0.968828
   learning_rate_0.04 0.113827 0.051946   0.047397   0.119878   0.174207      1.629526
       reg_lambda_3.0 0.113195 0.052327   0.047397   0.116767   0.175422      1.003664
       reg_lambda_1.5 0.111062 0.047102   0.047397   0.125924   0.159865      1.157061
       reg_lambda_2.5 0.110878 0.049534   0.047397   0.116966   0.168270      0.962594
       reg_lambda_2.0 0.110775 0.047013   0.047397   0.125065   0.159863      0.876628
        subsample_0.8 0.110256 0.046702   0.047397   0.124133   0.159239      1.158386
             baseline 0.110256 0.046702   0.047397   0.124133   0.159239      1.432870
  min_child_weight_10 0.110256 0.046702   0.047397   0.124133   0.159239      0.931466
 colsample_bytree_0.8 0.110256 0.046702   0.047397   0.124133   0.159239      1.127482
  min_child_weight_27 0.109473 0.045084   0.048628   0.123393   0.156397      0.856087
  min_child_weight_25 0.109463 0.045081   0.048628   0.123365   0.156397      0.905556
colsample_bytree_0.75 0.109168 0.045983   0.047397   0.122449   0.157657      0.950515
  min_child_weight_20 0.109106 0.044698   0.048628   0.123420   0.155271      1.229024
  min_child_weight_30 0.108748 0.044817   0.048628   0.121430   0.156186      0.859959
          max_depth_6 0.095460 0.043184   0.035841   0.113806   0.136734      1.209145
          max_depth_7 0.058735 0.039001   0.012104   0.107561   0.056541      1.532201


3 split rolling validation bc IC fluctuates across windows 

averaging across 3 windows gives more robust estimate
tracking std deviation shows which params are stable vs noisy 
max depth_3 std 0.0718 noisy
lr_0.03 0.0578 noisy
subsample_0.75 0.0411... more stable

should test param combos now bc tuning one param at a time misses interactions
test best param combo vs baseline vs other random combos 


=== Final Results ===
                              name       ic   ic_std  split1_ic  split2_ic  split3_ic  elapsed_time
combo_high_subsample_aggressive_d5 0.140208 0.046109   0.088805   0.131163   0.200658      0.971134
combo_high_subsample_aggressive_d4 0.139129 0.044276   0.092488   0.126269   0.198629      0.799622
                     final_depth_4 0.139129 0.044276   0.092488   0.126269   0.198629      1.014637
                      final_mcw_15 0.138225 0.045231   0.092661   0.122126   0.199888      0.710381
           combo_colsample_lr_deep 0.137457 0.045493   0.092661   0.119863   0.199848      0.721075
               final_locked_depth3 0.137457 0.045493   0.092661   0.119863   0.199848      0.733613
         combo_colsample_lr_deeper 0.132335 0.039858   0.085183   0.129164   0.182659      0.758095
                     final_reg_0.5 0.129274 0.033472   0.092661   0.121598   0.173565      0.887395
                       final_mcw_5 0.128336 0.032890   0.092661   0.120326   0.172020      0.977625
                     final_reg_1.5 0.127352 0.031973   0.092661   0.119581   0.169813      0.928629
               combo_high_sampling 0.126448 0.054206   0.058279   0.130164   0.190900      0.814115
                    combo_lower_lr 0.125336 0.056574   0.053955   0.129731   0.192323      0.952624
                   combo_stable_lr 0.124961 0.056052   0.053953   0.129949   0.190980      0.750607
                 combo_all_winners 0.124490 0.055420   0.053955   0.130164   0.189350      0.722097
               combo_max_colsample 0.124490 0.055420   0.053955   0.130164   0.189350      0.745591
                     final_depth_2 0.124256 0.066293   0.040738   0.129128   0.202902      1.092639
              combo_stable_focused 0.123535 0.054121   0.054516   0.129395   0.186696      0.797111
                  combo_ic_focused 0.121009 0.067580   0.039752   0.118067   0.205209      0.807101
                    combo_balanced 0.119241 0.063853   0.040519   0.120288   0.196915      0.827328
                     combo_high_lr 0.119123 0.051292   0.054516   0.122866   0.179987      0.696639
                          baseline 0.110256 0.046702   0.047397   0.124133   0.159239      0.591325
