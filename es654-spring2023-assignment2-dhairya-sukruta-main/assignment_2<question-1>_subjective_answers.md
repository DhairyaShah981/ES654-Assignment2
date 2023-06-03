# Plot
![image](https://user-images.githubusercontent.com/76472249/219765307-1b9ffebf-f34e-4307-89b7-b58bc39b6ead.png)
<br>
Here, we first bootstrap 1000 datasets from the original dataset with replacement, because we always find the bias and variance over models with different training sets which we form by bootstrapping. The bootstrap samples may contain repeated values because we are taking sample of size = len(x) but with replace =True. Although it won't affect the plots much because the repeated points will also be random in all the datasets.
<br>
## For a particular depth d:
### Bias is the deviation of our model (Expectation over all possible training datasets) from the true function.
![image](https://user-images.githubusercontent.com/76472249/219835486-128442a4-1a43-4e60-ac87-25cb3395ad68.png)

### Variance is the variance between different predictors on each sample averaged over all samples.
![image](https://user-images.githubusercontent.com/76472249/219835651-63cb9099-329a-460f-b59a-88083f0742e9.png)
<br>

If we plot the bias-variance tradeoff curve against increasing complexity of model (depth of the decision tree), we know that the bias will decrease and the variance will increase.
It is because as we go from low depth (high error on training set) to high depth the mean squared error decreases and thus the deviation from the true function decreases.
Although error on training set decreases, the model complexity increases and it starts predicting outliers also correctly which means all the predictors become very specific, and thus 
the variance between different predictors increases.
<br>
### Tradeoff between bias and variance: 
Variance is low when bias is high because model complexity (depth) is low i.e., all models behave similarly. The train and test accuracy are both less. (Underfitting)
<br>
Variance is high when bias is low because model complexity (depth) is high i.e., models become very specific. The train accuracy is very high, but the test accuracy decreases. (Overfitting)
<br>
There is a chance that bias and variance intersect, but it depends on the dataset. Right now we don't have a very sophisticated dataset and thus bias and variance are not intersecting.


