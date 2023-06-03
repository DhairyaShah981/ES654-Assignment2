# Question 2



<h2>Weighted Decision Trees - Plots</h2>
<p>Initially, we created a series of randomly distributed weights between 0 and 1 to train our decision tree. The following surface plots were obtained:</p>
<table>
  <tr>
    <td><strong>Our Weighted Decision Tree</strong><br>
      <img src="https://user-images.githubusercontent.com/76052389/220259814-fe9e31a0-602f-4d3f-ad96-d83a8675a11d.png" width="300" height="250">
    </td>
    <td><strong>Decision Tree without Weights</strong><br>
      <img src="https://user-images.githubusercontent.com/76052389/220260134-0764e5ee-f052-42ab-92db-e6b5bac72277.png" width="300" height="250">
    </td>
    <td><strong>Weighted Decision Tree from Sklearn</strong><br>
      <img src="https://user-images.githubusercontent.com/76052389/220260274-91e74581-1f26-4e83-b725-b9293c8799a6.png" width="300" height="250">
    </td>
  </tr>
</table>
<p>Next, we created a skewed weight series, with 20% of the weights much larger than the others, to test the accuracy of our decision tree.</p>
<table>
  <tr>
    <td><strong>Our Weighted Decision Tree</strong><br>
      <img src="https://user-images.githubusercontent.com/76052389/220260827-3cd03319-45c5-4f86-8ea5-849b8affc9f0.png" width="300" height="250">
    </td>
    <td><strong>Decision Tree without Weights</strong><br>
      <img src="https://user-images.githubusercontent.com/76052389/220260930-ada0f802-9c37-439a-b9cb-18204044644f.png" width="300" height="250">
    </td>
    <td><strong>Weighted Decision Tree from Sklearn</strong><br>
      <img src="https://user-images.githubusercontent.com/76052389/220260864-3ee63c18-fa0f-43e1-81da-02176a6d3299.png" width="300" height="250">
    </td>
  </tr>
</table>





<h2>Results</h2>

<p>Following are the accuracy results obtained from the dataset. They also include a column for the implimentation of the decision tree without shuffling the data: </p>

<table>
  <tr>
    <th>Type of tree</th>
    <th>Random weights</th>
    <th>Skewed weights</th>
  </tr>
  <tr>
    <td>Weighted decision tree(without shuffling the data)</td>
    <td>0.93</td>
    <td>0.93</td>
  </tr>
  <tr>
    <td>Weighted decision tree(shuffled data) </td>
    <td>0.77</td>
    <td>0.77</td>
  </tr>
  <tr>
    <td>Weighted decision tree (scikit-learn)</td>
    <td>0.83</td>
    <td>0.77</td>
  </tr>
</table>

<h2>Conclusions</h2>

<p> Firstly, it can be observed that, due to how the data is distributed, the decision tree can better predict the output when we do not stuffle the data. Now, after stuffling the data and assigning random weights, we can observe that the decision boundry obtained from the classifiers without the weights as inputs are different from those obtained without giving weighted input points. Although the accuracy of the predicted does not improve that much since the weights are asssigned randomly without taking into account how the data is distributed. The accuracy of our decision tree is very close to that of sklearns in the first case, but is a little lesser when it comes to the second case. However, the decision boundaries are very similar to each other only differing at few points. 
  
Thus the accuracy of the output largely depends on how the weights are distributed. Overall, the experiments show that using weighted decision trees can be more effective than traditional decision trees when the data is imbalanced or when certain data points are more important than others. However, the effectiveness of weighted decision trees heavily depends on how the weights are assigned and the distribution of data.

In conclusion, while using weighted decision trees may result in improved accuracy, care must be taken while assigning weights to the data points. In cases where certain data points are more important, assigning appropriate weights can lead to more effective decision making. </p> 







