import pandas as pd

file_path = 'C:/Users/MRX/Downloads/Data_Puskesmas.xlsx'
all_sheets = pd.read_excel(file_path, sheet_name=None)

print('Output Dict of DataFrames:')
for sheet_name, df in all_sheets.items():
    print(f'Sheet Name: {sheet_name}')
    print(df)
    print('\n')
