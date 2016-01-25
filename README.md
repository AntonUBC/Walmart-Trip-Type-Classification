## My solution to [Walmart Trip Type Classification challenge] (https://www.kaggle.com/c/walmart-recruiting-trip-type-classification)

This is the strongest model from my final ensemble. This model gives the log loss score 0.589 and approximately 59th place on the leaderboard (top 5.5%).

Here is the graphical representation of the solution:
!(https://github.com/AntonUBC/Walmart-Trip-Type-Classification/blob/master/pictures/Chart-1.png)















Below, I will go over each of the stages in more detail. However, what is interesting about this solution is that the ensemble stage models (NN and GBT) are not trained at all! I simply did not have time to train them before the deadline since I entered this competition a week before the deadline (I used only 3 submissions overall). My strategy was to choose reasonable parameters together with relatively small learning rates and determine the number of epochs for NN classifier and the number of trees for GBT classifier using early stopping. I will probably never find it out but it seems that appropriate training at the ensembling stage could push the LB score of this model significantly further. Thus, the relatively high score of this model can be attributed to three crucial elements:
1. Extensive feature engineering
2. Stacked generalization
3. Amazing python libraries XGBoost (GBT) and Keras (NN) which produce good results even with minimum amount of training
