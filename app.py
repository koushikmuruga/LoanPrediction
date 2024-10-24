import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from flask import Flask,render_template,request
from source.pipeline.predict_pipeline import predictPipeline,getData

app=Flask(__name__)

@app.route('/',methods=['GET','POST'])
def fn():
    if request.method=='GET':
        return render_template('index.html')
    else:
        data=getData(
            Gender=request.form.get('gender'),
            Married=request.form.get('married'),
            Dependents=request.form.get('dependents'),
            Education=request.form.get('education'),
            Self_Employed=request.form.get('self_employed'),
            LoanAmount=float(request.form.get('loan_amount')),
            Loan_Amount_Term=float(request.form.get('loan_amount_term')),
            Credit_History=request.form.get('credit_history'),
            Property_Area=request.form.get('property_area'),
            ApplicantIncome=float(request.form.get('applicant_income')),
            CoapplicantIncome=float(request.form.get('coapplicant_income'))
        )

        data_to_pred=data.data_as_DF()

        pred=predictPipeline()
        pred_value=pred.predict_data(data_to_pred)

        return render_template('index.html',results=pred_value[0])


if __name__=='__main__':
    app.run(debug=True,host='0.0.0.0',port=8080)