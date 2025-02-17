import pandas as pd

# Path ke file Excel
file_path = 'C:/Users/MRX/Downloads/Data_Puskesmas_Merged.xlsx'
output_path = 'C:/Users/MRX/Downloads/Data_Analisis.xlsx'

# Membaca semua sheet dalam file
all_sheets = pd.read_excel(file_path, sheet_name=None)

# Dictionary untuk menyimpan hasil
summary_data = {}  # Menyimpan data tertinggi & terendah
remaining_data = {}  # Menyimpan data sisanya

# Loop melalui setiap sheet dalam Excel
for sheet_name, df in all_sheets.items():
    # Mengambil hanya kolom numerik
    numeric_df = df.select_dtypes(include=['number'])

    # List untuk menyimpan data hasil tertinggi & terendah
    result_list = []

    # Menemukan nilai tertinggi dan terendah di setiap kolom
    for col in numeric_df.columns:
        max_value = numeric_df[col].max()
        min_value = numeric_df[col].min()

        # Baris dengan nilai tertinggi
        max_row = df[df[col] == max_value].copy()
        max_row["Keterangan"] = f"Data Tertinggi - {col}"

        # Baris dengan nilai terendah
        min_row = df[df[col] == min_value].copy()
        min_row["Keterangan"] = f"Data Terendah - {col}"

        # Menyimpan hasil ke list
        result_list.append(max_row)
        result_list.append(min_row)

    # Menggabungkan semua hasil ke satu dataframe
    if result_list:
        summary_data[sheet_name + " - Tertinggi & Terendah"] = pd.concat(result_list, ignore_index=True)

    # Menyimpan data sisanya (yang bukan tertinggi atau terendah)
    rows_to_exclude = pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    remaining_df = df.loc[~df.index.isin(rows_to_exclude.index)].copy()

    # Jika masih ada sisa data, simpan ke sheet berbeda
    if not remaining_df.empty:
        remaining_data[sheet_name + " - Data Lainnya"] = remaining_df

# Menyimpan ke file Excel baru dalam sheet yang terpisah
with pd.ExcelWriter(output_path) as writer:
    for sheet, data in summary_data.items():
        data.to_excel(writer, sheet_name=sheet, index=False)

    for sheet, data in remaining_data.items():
        data.to_excel(writer, sheet_name=sheet, index=False)

print(f"Hasil telah disimpan ke {output_path}")
