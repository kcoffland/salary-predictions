
# Salary Predictions Based on Job Descriptions

# Step 1: Define

### Overall Problem and Motivation

The goal is to produce a model which utilizes job postings to make a prediction on the salaries which should be given for that specific posting. This is valuable information for a company to accurately produce because it can help companies: 
* budget for new talent effectively
* utilize employees' time with more important projects
* give the company insight on what our competitors seem to be offering, so that we can adjust our offers appropriately

The information could also be used by a job board site such as Glassdoor to give applicants a better look as to which job postings would be the best fit. Companies are not obligated to give salary information even though it would attract more qualified candidates to the posting. Glassdoor gives a fantastic description as to why they estimate salaries [here](https://help.glassdoor.com/article/What-are-Salary-Estimates-in-Job-Listings/en_US/).

### The Data

The data will be read in from three CSV files:
1. __train_features.csv__: holds the feature data for each posting that will be used for training the model
2. __train_salaries.csv__: holds the target values for each posting that will be used for training the model
3. __test_features.csv__: which holds the feature data for the postings which will be used for my model's final predictions

The features being used to predict the salaries are job type, degree, major, industry, years of experience, and miles from metropolis. Each row is identified by a unique key named *jobId*. The target salary is given in units of thousands of dollars and are also identified by the unique key of *jobId*.


### Evaluation Metric

I will be evaluating model accuracy using mean squared error (MSE) between the predicted values and the actual target values. Since the testing target data is not given, I will be withholding a random chunk of the training data at the very beginning of the project to evaluate my model on. 

MSE is appropriate for regression models which penalizes larger errors, both over and under estimations, more severely than just using absolute error.

### Output

The final output will be a CSV file named *test_salaries.csv*. The format of this CSV will mirror the format of *train_salaries.csv* with the first column representing the *jobId* and the second column being the corresponding predicted salary in thousands of dollars. I will also export the final pipeline so that it could be utilized in the future without needing to retrain it.
