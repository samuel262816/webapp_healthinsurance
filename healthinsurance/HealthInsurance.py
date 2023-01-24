import pickle
import pandas as pd
import numpy as np
# from sklearn.preprocessing  import StandardScaler, MinMaxScaler

class HealthInsurance():
    
    def __init__( self ):

        self.annual_premium_scaler            = pickle.load( open( 'parameter/annual_premium_scaler.pkl', 'rb') )
        self.age_scaler                       = pickle.load( open( 'parameter/age_scaler.pkl', 'rb') )
        self.vintage_scaler                   = pickle.load( open( 'parameter/vintage_scaler.pkl', 'rb') )
        self.target_encode_region_code_scaler = pickle.load( open( 'parameter/target_encode_region_code_scaler.pkl', 'rb'))
        self.target_encode_gender_scaler      = pickle.load( open( 'parameter/target_encode_gender_scaler.pkl', 'rb'))
        self.fe_policy_sales_channel_scaler   = pickle.load( open( 'parameter/fe_policy_sales_channel_scaler.pkl', 'rb'))
        
        
    def data_cleaning( self, df1 ):
        new_cols = ['id', 'gender', 'age', 'region_code', 'policy_sales_channel', 'driving_license', 'vehicle_age', 
                    'vehicle_damage', 'previously_insured', 'annual_premium', 'vintage', 'response']
        
        df1.columns = new_cols
        
        return df1
        
        
    def feature_engineering( self, df2 ):
        # vehicle age
        df2['vehicle_age'] = df2['vehicle_age'].apply( lambda x: 'over_2_years' if x == '> 2 Years' else 'between_1_2_years' if x == '1-2 Year' else 'below_1_year')

        # vehicle damage
        df2['vehicle_damage'] = df2['vehicle_damage'].apply( lambda x: 1 if x == 'Yes' else 0)
        
        return df2
    

    def data_preparation( self, df5 ):
        # annual_premium
        df5['annual_premium'] = self.annual_premium_scaler.transform( df5[['annual_premium']].values )

        ### 5.2. Rescaling
        # age
        df5['age'] = self.age_scaler.transform( df5[['age']].values )

        # vintage
        df5['vintage'] = self.vintage_scaler.transform( df5[['vintage']].values )
        
        ### 5.3. Encoding
        # gender - target encoding
        df5.loc[: , 'gender'] = df5['gender'].map( self.target_encode_gender_scaler )
        
        # region_code - target encoding
        df5.loc[:, 'region_code'] = df5['region_code'].map( self.target_encode_region_code_scaler )

        # vehicle_age - one hot encoding
        df5 = pd.get_dummies( df5, prefix= 'vehicle_age', columns=['vehicle_age'] )       

        # policy_sales_channel - frequency encoding
        df5.loc[:, 'policy_sales_channel'] = df5['policy_sales_channel'].map( self.fe_policy_sales_channel_scaler )

        # features selections
        cols_selected = [ 'vintage', 'annual_premium', 'age', 'region_code', 'vehicle_damage', 'policy_sales_channel', 'previously_insured' ]
        
        return df5[ cols_selected ]
    
    
    def get_predict( self, model, original_data, test_data ):
        # model prediction
        pred = model.predict_proba( test_data )
        
        # join prediction to original data
        original_data['prediction'] = pred
        
        return original_data.to_json( orient= 'records', date_format= 'iso')