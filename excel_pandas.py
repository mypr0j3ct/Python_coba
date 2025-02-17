import pandas as pd

# Path file Excel
file_path = 'C:/Users/MRX/Downloads/Book1.xlsx'
output_file_path = 'C:/Users/MRX/Downloads/Book1_Modified.xlsx'

# Membaca semua sheet dalam file Excel
all_sheets = pd.read_excel(file_path, sheet_name=None)

# Menyiapkan writer untuk menulis file Excel baru
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    for sheet_name, df in all_sheets.items():
        # Indeks yang akan dipisahkan
        indices_to_separate = [3, 9, 12, 13, 21, 22, 34, 37]
        
        # Pisahkan dataframe berdasarkan indeks
        df_separated = df.loc[df.index.isin(indices_to_separate)]
        df_remaining = df.loc[~df.index.isin(indices_to_separate)]

        # Simpan ke dalam dua sheet yang berbeda
        df_separated.to_excel(writer, sheet_name=f"{sheet_name}_Terpisah", index=False)
        df_remaining.to_excel(writer, sheet_name=f"{sheet_name}_Sisa", index=False)

print(f"File telah berhasil disimpan di {output_file_path}")
