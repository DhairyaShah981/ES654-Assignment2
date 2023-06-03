# Question 5 (a)
We are using our own implementation of Decision Tree as the base estimator. According to RF algorithm we have modified our Decision Tree such that when we initialise Decision Tree in our Test Case, we need to write forest=1 and the value of m if we want to use Decision Tree for Random forest. If forest = 1 then in our Decision Tree we sample m features out of M at each node and then follow the usual ID3 algorithm.
<br>
<br>
'm' is a hyperparameter here, and we can find the optimal value of m using cross validation. Although in our implementation we have taken m = root(M) which is generally a good approximation of m. Also, we have not considered the case of Discrete Input because in that case we actually drop the feature after using it as a split. To accomodate the case of Discrete input we need to use one hot encoding so that we do not drop that feature after using it as a split which is required by the RF algorithm.
<br>
<br>
For plotting decision surfaces we are using 2d contour plots so it is compulsory to use only 2 features for each plot. Now each estimator could be using more than 2 features as we are sampling m features at each node. So we can plot all pairs of features for each individual estimators which can be a mess. That's why we are plotting decision surfaces of individual estimators using 2 random features from the list of their selected_attributes. Even, for combined estimator we can plot C(M, 2) number of plots between all pairs of features. So we again plot combined estimator using 2 random features.
<br>
### Random Forest Classifier
#### Output
![image](https://user-images.githubusercontent.com/76472249/220280736-a766b5f0-d2d1-4daf-abd1-14eb91f5e566.png)
  
#### Plotting Learnt Decision Trees
We have printed and written the learnt trees in .txt files which are uploaded in the plots folder.
#### Decision Surfaces of Individual Estimators
![image](https://user-images.githubusercontent.com/76472249/220279446-d3730c42-6f68-4a96-9f9d-5e4c55e72904.png)

#### Decision Surfaces of Combined Estimator
![image](https://user-images.githubusercontent.com/76472249/220279587-b30aa152-b1e4-4120-af91-992c1cc89929.png)

### Random Forest Regressor
For Random forest Regressor, it is not possible to show decision surfaces like we did in classification problem. Therefore here we take any two random features for all estimators and plot 3d decision surfaces for all the estimators as well as for combined estimators. X-Y plane represents the two random features and Z represents the prediction of the Random Forest Regressor.

#### Output
![image](https://user-images.githubusercontent.com/76472249/220263597-c6145a85-47da-4ba1-8c83-3dc63a87ea64.png)
#### Plotting Learnt Decision Trees
We have printed and written the learnt trees in .txt files which are uploaded in the plots folder.
#### Decision Surfaces of Individual Estimators
![image](https://user-images.githubusercontent.com/76472249/220304924-bef3532b-7f20-4dc9-aba0-211f61e86ce5.png)

#### Decision Surfaces of Combined Estimator
![image](https://user-images.githubusercontent.com/76472249/220305073-da285268-0772-41a6-8f40-c5280185f805.png)

# Question 5 (b)
### Random Forest Classifier
#### Output
![image](https://user-images.githubusercontent.com/76472249/220280929-fb4e9003-092a-431b-8a2f-403ec853f5a2.png)

#### Plotting Learnt Decision Trees
We have printed and written the learnt trees in .txt files which are uploaded in the plots folder.
#### Decision Surfaces of Individual Estimators
![image](https://user-images.githubusercontent.com/76472249/220283838-707286e5-01c1-4cc8-8117-1b8a8c7faed7.png)
#### Decision Surfaces of Combined Estimator
![image](https://user-images.githubusercontent.com/76472249/220283979-fdfd329a-3ff9-4de2-927d-1539ef7d9969.png)

