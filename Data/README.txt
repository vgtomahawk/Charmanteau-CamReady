The following files are included in this directory.

1. dataset.csv (Main dataset)
This is the main dataset. Each line in this dataset is of the form
y,x_1,x_2,flag
where
- y is the portmanteau (ground-truth)
- x_1 and x_2 are the root words (in that order)
- flag=knight or flag=other
  - if flag=knight, the examples belongs to D_Wiki, the dataset shared by Deri and Knight (2015). There are 401 such examples
  - if flag=other, the examples belong to D_Blind, there are 1223 such examples
Together, our dataset has 1624 examples

2. baselineResults.txt (Baseline Results)
These are the predictions of the BASELINE of Deri and Knight (2015) on the 1223 examples from D_Blind
Each line is of the form x_1 x_2 y_baseline, where
- x_1 and x_2 are the root words
- y_baseline is the prediction of the baseline model


3. best_blind_forward.txt. best_blind_backward.txt
Each file contains a list of examples. For each example, we have a rank-list (one candidate per line) in ascending order of log-loss (lesser loss is better). The top-most element in this list is the word output by our model. Finally the original word (\textsc{GROUND-TRUTH}) and the word output by our model are printed out.
Example: 
fashism: 0.0057
fasscism: 6.344
.
.
.
.
fashioascism: 39.074
Original Word: fashism
Output Word: fashism

The Original Word is the ground truth and the Output Word is the model prediction. The Output Word is the model prediction

best_blind_forward.txt corresponds to our best performing Forward model 
best_blind_backward.txt corresponds to our best performing Backward model
