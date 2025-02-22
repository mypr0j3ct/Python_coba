import pandas as pd

# Membaca file Excel
input_path = r'C:\Users\MRX\Downloads\Data_Puskesmas_Merged.xlsx'
sheet_name = 'MKN'
df = pd.read_excel(input_path, sheet_name=sheet_name)

# Memfilter data
filtered_df = df[(df['RHR'] >= 60) & (df['RHR'] <= 100)]  # Data valid (60-100)
outlier_df = df[(df['RHR'] < 60) | (df['RHR'] > 100)]  # Data di bawah 60 atau di atas 100

# Membuat file Excel baru dengan dua sheet
output_path = r'C:\Users\MRX\Videos\Captures\Python_Final\coba_data.xlsx'

with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    filtered_df.to_excel(writer, sheet_name='Filtered_Data', index=False, columns=['RHR', 'usia', 'ir', 'glu', 'chol', 'acd'])
    outlier_df.to_excel(writer, sheet_name='Outlier_Data', index=False, columns=['RHR', 'usia', 'ir', 'glu', 'chol', 'acd'])

# Menampilkan pesan konfirmasi
print("File Excel baru telah dibuat di:", output_path)
print("Sheet 'Filtered_Data' berisi data dengan RHR antara 60-100.")
print("Sheet 'Outlier_Data' berisi data dengan RHR di bawah 60 atau di atas 100.")
