# Loan-Prediction-Analysis-
Dream Housing Finance company deals in all home loans. They have presence across all urban, semi urban and rural areas. Customer first apply for home loan after that company validates the customer eligibility for loan. Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have given a problem to identify the customers segments, those are eligible for loan amount so that they can specifically target these customers.  This is a standard supervised classification task.A classification problem where we have to predict whether a loan would be approved or not. Below is the dataset attributes with description.


# How to start

The second way is to manually clone this repository and change it later by own. Project is ready to run (with some requirements). You need to clone and run:


```bash
$ mkdir Project
$ cd Project
$ git@github.com:wiahwajith/Loan-Prediction-Analysis-.git
$ make
$ make run
```

Open http://127.0.0.1:5000/, customize project files and have fun.

If you have any ideas about how to improve it Fork project and send me a pull request.

# Project structure

```bash
├── .vscode
├── colab
      └── Loan_Prediction_Analysis_Classification.ipynb
├── model
     └── savemodel.sav
├── static
├── tempaltes
      ├── index.html
      ├── passResult.html
      ├── predictorform.html
      └── failResult.html
├── README.md
├── app.py
├── Loan Prediction Dataset.csv
├── Procfile
├── requirements.txt
├── runtime.txt

```

# Requirements

python 3.7.12

# packages

•	Pandas
•	Numpy
•	Seaborn
•	Matplotlib
•	SKlearn
•	Xgboost
•	Pickle
•	Flask

