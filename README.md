# ML_Midterm

Python libraries used: numpy, matplotlib.pyplot, pandas, sklearn, joblib, 

Task 1

Analyze the Pearson formula and implement it in Python to calculate the correlation coefficient. The code can be found in the file: Correlation.py. 

The coefficient obtained is -0.411. 

After performing the calculation, we generated the graph file:graph1.png.

Task 2

I analyzed and uploaded a CSV file named "a_latsuzbaia24_563892." 

I attempted to train a model using this data, splitting the dataset into 70% for training and 30% for testing to maintain class balance. I utilized the Scikit-learn library in Python, incorporating features such as StandardScaler, train_test_split, and Logistic Regression with default settings. 

The training script is saved as "model.py," which generated a joblib file named "model.joblib" along with the model coefficients.

Intercept - 1.71668275
"words" - 1.69551979
"links" - 2.17128522
"capital_words" - 3.18518512
"spam_word_count" - 2.40954686

To evaluate email text, I created a script called "spam_classifier.py." This code checks for specific keywords; if an email contains any of these words, it is classified as spam. Additionally, users can manually input any data for testing purposes.
