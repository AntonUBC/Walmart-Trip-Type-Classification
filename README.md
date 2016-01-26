## Solution to [Walmart Trip Type Classification Challenge] (https://www.kaggle.com/c/walmart-recruiting-trip-type-classification)

This is the strongest model from my final ensemble. This model gives the log loss score 0.589 and approximately 59th place on the leaderboard (top 5.5%).

Here is the graphical representation of the solution:
![alt tag](https://github.com/AntonUBC/Walmart-Trip-Type-Classification/blob/master/pictures/Chart-1.png)

Below, I will go over each of the stages in more detail. However, what is interesting about this solution is that the ensemble stage models (NN and GBT) are not trained at all! I simply did not have time to train them since I entered this competition a week before the deadline (I used only 3 submissions overall). My strategy was to choose reasonable parameters together with relatively small learning rates and determine the number of epochs for NN classifier and the number of trees for GBT classifier using early stopping. I will probably never find it out but it seems that appropriate training at the ensembling stage could push the LB score of this model significantly further. Thus, the relatively high score of this model can be attributed to three crucial elements:
1. Extensive feature engineering
2. Stacked generalization
3. Amazing python libraries XGBoost (GBT) and Keras (NN) which produce good results even with minimum amount of training

### Project Description

The task was to classify customer trips using only a transactional dataset of the items they have purchased. Walmart has categorized the trips contained in the data into 38 distinct types using a proprietary method applied to an extended set of data. Contestants were challenged to recreate this categorization/clustering with a more limited set of features. This could provide new and more robust ways to categorize trips. The submitted solution had to be in the form of matrix of predicted probabilities with dimensions (# of customers, # of trip types).

### Data

Data fields:

- TripType - a categorical id representing the type of shopping trip the customer made. This is the ground truth that we had to predict
- VisitNumber - an id corresponding to a single trip by a single customer
- Weekday - the weekday of the trip
- Upc - the UPC number of the product purchased
- ScanCount - the number of the given item that was purchased. A negative value indicates a product return.
- DepartmentDescription - a high-level description of the item's department
- FinelineNumber - a more refined category for each of the products, created by Walmart

Number of rows in the training dataset: 647,054. Number of rows in the test dataset: 653,646

The data was multi-dimensional (several rows per customer). However, the submission format assumed only one row of predicted probabilities per customer. Therefore, the dataset must have been transformed from long to wide format, which created enormous opportunities for feature engineering.
 
### Feature Engineering

I created the following features from the datafields described above:

- Min, Max, Mean, and Range = Max-Min of the number of products per Visit-Department combination
- Indicators for missing department and finelinenumber information
- Indicator for product return
- Number of Upc products, Fineline products, and Departments visited
- Sum of products bought
- Ratios: #Fineline/#Departments, #Upc/#Departments, mean/min, max/mean
- Product counts for 70% most frequent Fineline products
- Product counts by Department
- Number of returned products by department
- Number of multiproduct purchases by department
- Counts of multiple rows by department
- Counts of rows excluding product returns by department
- Counts of rows with product return by department
- Department dummies and their two-way interactions (only for 60% most frequent departments)

Dimensions of resulting data: 95,674 X 4,490

### Stacked Generalization

[Stacked generalization](http://machine-learning.martinsewell.com/ensembles/stacking/) is an ensembling algorithm which allows combining predictions of multiple models by estimating optimal ensemble weights for each prediction. Here, by optimal I mean weights which give the highest cross-validation score. In the simplest case, stacking can be performed with a linear regression. In general, using more sophisticated learning algorithms can significantly improve model performance. To reduce overfitting, stacking of training data is performed using out-of-fold predictions which is shown in the following figure:

![alt tag](https://github.com/AntonUBC/Walmart-Trip-Type-Classification/blob/master/pictures/Chart-2.png)

Here, training data is split into three folders. What we have to do is just recursively train each of our models on the two folders and generate predictions for the third folder. Finally, vertically stacking all three prediction vectors (or matrices), we obtain our stacked training data. Note, that the stacking occurs in the horizontal dimension too, since we adding predictions of different models. In general, the more folders the data are split into, the less is the risk of overfitting (5-10 folders is the optimal number similarly to the usual CV). Stacking test data is performed in the conventional way.

In this competition I used 15 different models in the stacking procedure:

- KNN with 10, 20, 40, 80, 120, and 160 neighbors
- Gradient Boosting Trees
- AdaBoost with CART
- Random Forest
- Extra Trees
- Feed-forward NN with two hidden layers (1024 units each) estimated using the RMSPROP algorithm
- Feed-forward NN with two hidden layers (1024 units each) estimated using the AdaGrad algorithm
- Feed-forward NN with three hidden layers (512 units each) estimated using the RMSPROP algorithm
- Feed-forward NN with three hidden layers (512 units each) estimated using the AdaGrad algorithm
- Logistic regression

To reduce memory burden, tree models were trained using sparse data (thanks to XGBoost and Scikit-Learn!).
Neural nets were trained on dense matrices using [PReLU activation](http://arxiv.org/abs/1502.01852), [batch normalization](http://arxiv.org/abs/1502.03167), and dropout for regularization (thanks to Keras again!). Training neural nets on such big data would be practically impossible without GPU (thanks Theano!). My average Nvidia-Quadro performed one epoch of training in impressive 10-20 sec. (depending on the batch size).







