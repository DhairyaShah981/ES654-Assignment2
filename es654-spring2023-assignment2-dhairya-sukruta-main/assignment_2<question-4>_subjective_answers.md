# Question 4 (a)

### Output
![image](https://user-images.githubusercontent.com/76472249/219847115-8d29f07c-eb81-42a8-95ad-a2f003bcc649.png)

### Decision Surfaces of Individual Estimators
![image](https://user-images.githubusercontent.com/76472249/220274567-9423a22b-6f6c-49c2-9a51-eacd528c84f1.png)
### Decsion Surface of Combined Estimator
![image](https://user-images.githubusercontent.com/76472249/220274595-053f05b4-d34f-42a4-b77c-8c56701a6819.png)
### Implementation
We grow all the estimators to max depth and while plotting we take any random two features from the featureSet and then plot all the individual estimators and the combined estimators on the basis of those two features although the test case has only 2 features. 

# Question 4 (b)
To implement bagging in parallel fashion, we introduced n_jobs in the __init__ function whose default value is -1 which means it uses all available CPUs for parallel execution. n_estimators = 3

### Output when n_jobs = 1
![image](https://user-images.githubusercontent.com/76472249/219851250-d897d51a-d51a-4fb1-b1f7-9dd96606d7ab.png)

### Output when n_jobs = -1
![image](https://user-images.githubusercontent.com/76472249/219851267-fd80c9ab-1364-42df-96d3-7f52edd7bde8.png)

### Insights
The above result is very interesting. When n_jobs = -1, it uses all available CPUs for parallel execution. Although time taken for the code to execute is very high than time taken when it uses only 1 CPU for execution (n_jobs=1). In some cases, setting n_jobs to -1 (i.e., using all available CPU cores) may actually slow down the computation due to overheads associated with parallelization, especially if the dataset is small. In other cases, setting n_jobs to a lower value may provide better performance, especially if the dataset is large and/or the model is computationally expensive.
<br>
<br>

### Plot between time and n_jobs
![image](https://user-images.githubusercontent.com/76472249/219851720-10ef4c7a-2403-4690-9852-705d93b0c081.png)<br>
Using this plot we can select the optimum value of n_jobs for a particular dataset and a particular value of n_estimator.
