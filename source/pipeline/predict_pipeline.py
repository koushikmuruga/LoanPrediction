import sys
import pandas as pd
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from exception import CustomException
from utils import load_object
from logger import logging

class predictPipeline:
    def __init__(self):
        pass

    def predict_data(self,data_df):

        logging.info('Loading Pickle files')
        model_path=os.path.join('Data_Folder','model.pkl')
        preprocessor_path=os.path.join('Data_Folder','preprocessor.pkl')

        model=load_object(path=model_path)
        preprocessor=load_object(path=preprocessor_path)

        data=preprocessor.transform(data_df)
        pred_value=model.predict(data)

        logging.info(f'Predicted value: {pred_value[0]}')

        return pred_value

class getData:
    def __init__(self,
                 Gender: str,
                 Married: str,
                 Dependents: str,
                 Education: str,
                 Self_Employed: str,
                 LoanAmount: float,
                 Loan_Amount_Term: float,
                 Credit_History: str,
                 Property_Area: float,
                 ApplicantIncome: float,
                 CoapplicantIncome: float):
        
        self.Gender = Gender

        self.Married = Married

        self.Dependents = Dependents

        self.Education = Education

        self.Self_Employed = Self_Employed

        self.LoanAmount = LoanAmount

        self.Loan_Amount_Term = Loan_Amount_Term

        self.Credit_History = Credit_History

        self.Property_Area = Property_Area

        self.Total_Income = ApplicantIncome + CoapplicantIncome

    def data_as_DF(self):
        
        try:
            dict={"Gender": [self.Gender],
                "Married": [self.Married],
                "Dependents": [self.Dependents],
                "Education": [self.Education],
                "Self_Employed": [self.Self_Employed],
                "LoanAmount": [self.LoanAmount],
                "Loan_Amount_Term": [self.Loan_Amount_Term],
                "Credit_History" : [self.Credit_History],
                "Property_Area" : [self.Property_Area],
                "Total_Income" : [self.Total_Income]
                }
            
            logging.info('Input data converted to DataFrame')
            
            return pd.DataFrame(dict)
        
        except Exception as e:
            raise CustomException(e,sys)


    
