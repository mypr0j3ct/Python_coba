import pandas as pd

# Path file Excel
file_path = 'C:/Users/MRX/Downloads/Data_Puskesmas.xlsx'

# Membaca semua sheet dalam file Excel
all_sheets = pd.read_excel(file_path, sheet_name=None)

# List untuk menyimpan DataFrame dari setiap sheet
df_list = []

# Iterasi melalui setiap sheet
for sheet_name, df in all_sheets.items():
    # Standarisasi nama kolom agar seragam
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Normalisasi kolom HR karena ada yang "HR (Bpm)" dan "HR (bpm)"
    hr_columns = [col for col in df.columns if 'hr' in col]
    if len(hr_columns) == 1:
        df.rename(columns={hr_columns[0]: 'hr'}, inplace=True)

    # Menormalkan kolom-kolom yang seharusnya ada di semua sheet
    expected_columns = ['ir', 'usia', 'hr', 'glu', 'chol', 'acd']
    for col in expected_columns:
        if col not in df.columns:
            df[col] = None  # Menambahkan kolom yang hilang dengan nilai None

    # Pilih hanya kolom yang diinginkan
    df = df[expected_columns]

    # Tambahkan ke daftar DataFrame
    df_list.append(df)

# Menggabungkan semua DataFrame menjadi satu
merged_df = pd.concat(df_list, ignore_index=True)

# Simpan hasil ke file Excel
output_file = 'C:/Users/MRX/Downloads/Data_Puskesmas_Merged.xlsx'
merged_df.to_excel(output_file, index=False)

print(f"File hasil penggabungan telah disimpan di: {output_file}")

# Opsional: Menampilkan beberapa baris pertama di terminal
print(merged_df.head())
