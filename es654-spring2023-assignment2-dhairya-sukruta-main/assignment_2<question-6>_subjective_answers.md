# Gradient Boosting

### Output
![image](https://user-images.githubusercontent.com/76472249/220287041-cc8df514-f6c3-4359-b736-c59eb5f5ad98.png)
### The predictions
![image](https://user-images.githubusercontent.com/76472249/220287120-280dcc6e-f8c2-42e1-ba34-296b5a675be9.png)

The RMSE value that we obatined after using the gradient boost classifier to predict the output was around 53. We used our own decision tree with max depth set to 10 inorder to train the data and used default values for the rest of the hyperparameters. In the case of the given dataset, the noise parameter is set to 10, which is quite high. This implies that there is a lot of randomness in the data, making it difficult for the model to fit a reliable pattern. In such cases, Gradient Boosting may not be the best choice of algorithm, as it is highly susceptible to overfitting when the noise levels in the data are high. It is possible that the model is trying to fit too closely to the training data, leading to poor generalization to new data. Thus, in the future, in order to implove the accuracy of the model, we could redoce the complexity of the decision trees by reducing the max depth value further. We could alos have to tune the other hyperparameters in order to obtain the best possible output. 
