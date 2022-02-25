
from flask import Flask ,render_template , request
import pickle
import numpy as np


app = Flask(__name__)

#load model 
loan_model = pickle.load(open('models/saveloanmodel.sav' , 'rb'))

@app.route("/")
def hello():
    return render_template ("index.html")

@app.route("/predict", methods = ['POST'])
def submit():
    #html -> .py
    if request.method == "POST":

        name = request.form["username"]
        Gender = request.form["Gender"]
        Married = request.form["Married"]
        Dependents = request.form["Dependents"]
        Education = request.form["Education"]
        Self_Employed = request.form["Self_Employed"]
        Credit_History = request.form["Credit_History"]
        Property_Area = request.form["Property_Area"]

        Total_Income_Log = np.log(float(request.form["ApplicantIncome"] + request.form["CoapplicantIncome"]) )
        Loan_Amount_Term_Log =  np.log( float(request.form["Loan_Amount_Term_Log"]) )
        LoanAmountLog =  np.log( float(request.form["LoanAmountLog"]) / 1000)
        ApplicantIncomeLog =   np.log( float(request.form["ApplicantIncome"]) ) 
 
      
       

        result = loan_model.predict([[ Gender,	Married,	Dependents,	Education,	Self_Employed,	Credit_History,	Property_Area,	ApplicantIncomeLog,	LoanAmountLog,	Loan_Amount_Term_Log,	Total_Income_Log]])

    #.py -> HTML
    return render_template ("index.html",**locals())



if __name__ == "__main__":
    app.run(debug=True)