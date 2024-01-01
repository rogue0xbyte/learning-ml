import pandas as pd
import numpy as np

def preprocess(data):

	data['In_Game_Price'] = data['In_Game_Price'].replace(',', '', regex=True).astype(float)

	data['stock_specs'] = data['stock_specs'].replace('??', 0)
	data['Drive_Type'] = data['Drive_Type'].replace('??', 0)
	data['speed'] = data['speed'].replace('??', 0)
	data['handling'] = data['handling'].replace('??', 0)
	data['acceleration'] = data['acceleration'].replace('??', 0)
	data['launch'] = data['launch'].replace('??', 0)
	data['braking'] = data['braking'].replace('??', 0)
	data['Offroad'] = data['Offroad'].replace('??', 0)
	data['Top_Speed'] = data['Top_Speed'].replace('??', 0)
	data['0-60_Mph'] = data['0-60_Mph'].replace('??', 0)
	data['g-force'] = data['g-force'].replace('??', 0)
	data['Horse_Power'] = data['Horse_Power'].replace('??', 0)
	data['Weight_lbs'] = data['Weight_lbs'].replace('??', 0)

	data['speed'] = data['speed'].replace('info_not_found', 0)
	data['handling'] = data['handling'].replace('info_not_found', 0)
	data['acceleration'] = data['acceleration'].replace('info_not_found', 0)
	data['launch'] = data['launch'].replace('info_not_found', 0)
	data['braking'] = data['braking'].replace('info_not_found', 0)
	data['Offroad'] = data['Offroad'].replace('info_not_found', 0)
	data['Top_Speed'] = data['Top_Speed'].replace('info_not_found', 0)
	data['0-60_Mph'] = data['0-60_Mph'].replace('info_not_found', 0)
	data['g-force'] = data['g-force'].replace('info_not_found', 0)
	data['Horse_Power'] = data['Horse_Power'].replace('info_not_found', 0)
	data['Weight_lbs'] = data['Weight_lbs'].replace('info_not_found', 0)

	data = pd.get_dummies(data, columns=['stock_specs'], prefix='stock')

	data = pd.get_dummies(data, columns=['Drive_Type'], prefix='drive')

	data['Top_Speed'] = data['Top_Speed'].replace('info_not_found', 0)
	data['Top_Speed'] = data['Top_Speed'].str.replace(' Mph', '').astype(float)

	data['0-60_Mph'] = data['0-60_Mph'].replace('info_not_found', 0)
	data['0-60_Mph'] = data['0-60_Mph'].str.replace('s', '').astype(float)

	data['g-force'] = data['g-force'].replace('info_not_found', 0)
	data['g-force'] = data['g-force'].str.replace(' g', '').astype(float)
	
	data['Weight_lbs'] = data['Weight_lbs'].replace(',', '', regex=True).astype(float)
	data['Horse_Power'] = data['Horse_Power'].replace(',', '', regex=True).astype(float)

	data = data.fillna(0)

	return data