import pandas as pd
import random

# Path file Excel
file_path = 'C:/Users/MRX/Downloads/Book1.xlsx'
output_file = 'C:/Users/MRX/Downloads/Split_Book1.xlsx'

# Membaca semua sheet dalam file Excel
all_sheets = pd.read_excel(file_path, sheet_name=None)

# Mengambil dataframe dari sheet pertama (misalnya 'Sheet1')
sheet_name = list(all_sheets.keys())[0]  # Ambil nama sheet pertama
df = all_sheets[sheet_name]  # Ambil dataframe dari sheet tersebut

# Mengacak indeks dataframe
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Memisahkan 7 dataframe pertama dan 25 dataframe lainnya
df_selected = shuffled_df.iloc[:7]
df_remaining = shuffled_df.iloc[7:]

# Menulis ke file Excel baru dengan dua sheet
with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    df_selected.to_excel(writer, sheet_name='Selected_7', index=False)
    df_remaining.to_excel(writer, sheet_name='Remaining_25', index=False)

print(f"File Excel baru berhasil disimpan di: {output_file}")
