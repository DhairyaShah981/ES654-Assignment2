# Question 3 (a) Real Input Discrete Output

We have used base_estimator as sklearn Decision Tree Classifier, and 3 estimators in our ensemble. As asked in the question, we have to implement ADAboost on Decision Stump (depth = 1 tree). 
We have a Real Input Discrete Output, binary class problem with 2 features in the feature set. 
<br>
### Output
![image](https://user-images.githubusercontent.com/76472249/219843670-6c544fd7-2aa0-45b9-90e0-393454fd0b2c.png)

### Decision Surfaces of Individual Estimators
![image](https://user-images.githubusercontent.com/76472249/219855082-0932fa4a-a3ce-47a7-b1b0-ccc0e8673409.png)

### Decision Surface of Combined Estimators
![image](https://user-images.githubusercontent.com/76472249/219855086-11eb7856-04df-418b-9ba7-75486169d50e.png)

<br>
For plotting the decision surfaces we are taking first two features. Although the test cases has only 2 features.

### Insights
The testcase given to us was a binary class problem so the prediction on the basis of sign would have also worked, but we implemented ADAboost such that it can also work for a multi-class problem where the output is non binary.
Our prediction looked at the weighted sum of alphas of all estimators and predicted the class with max sum. To test this we increased the classes to 3 but the accuracy dropped dramatically to less than random. (0.5) 
To debug we printed alphas, and some of them were negative. Then we realised inspite of having 3 classes, we are using depth = 1 trees which means individual trees are only worse than random and therefore the ensemble of them would also be worse than random.
To solve this we need to use base_estimator as depth = 2 trees. 
<br>
**Generalising**: For a multi-class problem, we need base_estimator with depth = (classes - 1) atleast to make our ensemble better than random.

# Question 3 (b) 
We now have a binary class - classification problem with 2 features in the feature set. We are trying two approaches:
<br>
### 1 Decision Stump (depth = 1 tree)
![image](https://user-images.githubusercontent.com/76472249/219844751-fbcb4608-67ff-469a-acc1-6ffe5db0b172.png)

### ADAboost of 3 depth = 1 trees
![image](https://user-images.githubusercontent.com/76472249/219844756-495f5505-9b98-4f17-8c77-ee0c518f64cf.png)
<br>

**Decision surface of a single sklearn decision stump** <br>
![image](https://user-images.githubusercontent.com/76472249/219845854-8bd55e06-825a-4d72-8485-c09001f76e75.png)

**Decision Surfaces of Individual Estimators** <br>
![image](https://user-images.githubusercontent.com/76472249/219855102-b293cb49-056f-4929-a403-9215156c51b0.png)

**Decision Surface of Combined Estimators**<br>
![image](https://user-images.githubusercontent.com/76472249/219855111-873e299b-f282-441e-ac4d-bb87664f2f4f.png)

### Insights 
On observing the dataset, we can observe that the dataset is easily clusterable and there are very less outliers. Also from the decision surface of decision stump and combined estimator we can see that both of them are doing mistakes on the same outliers.
Therefore the accuracy of both is coming out to be the same = 0.92. This means that even if we make an ensemble of large number of decision stumps, the accuracy will remain same. 
Now to increase the accuracy and correctly predict the outliers, we need to increase the depth of the decision trees.
Also, you can see that 1st estimator in the adaboost is the decision stump only, and its alpha is coming out to be the highest. Therefore the ensemble will behave mostly like the 1st estimator which is the decision stump only. Therefore this is one more reason why the accuracy of decision stump and ensemble is comping out to be exactly same.
<br>

One more thing we noticed while implementing ADAboost is that the alpha is not **strictly increasing** on further iterations. We thought that it should increase on further iterations because it depends inversely on error and as we are trying to decrease error, the alpha should increase. But after experimenting with few more datasets we realised that it depends on the specific characteristics of the dataset being used. For example, if the dataset is noisy or has a large amount of outliers, a smaller value of alpha may be more appropriate to reduce the impact of these difficult data points.
