import pandas as pd


# Fungsi umum untuk validasi kolom
def validate_columns(df, columns):
    """
    Memvalidasi apakah semua kolom ada dalam DataFrame dan bertipe numerik.
    Mengembalikan daftar kolom yang valid.
    """
    valid_columns = []
    for col in columns:
        if col not in df.columns:
            print(f"Error: Kolom '{col}' tidak ditemukan dalam DataFrame.")
        elif not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Error: Kolom '{col}' bukan tipe data numerik.")
        else:
            valid_columns.append(col)
    return valid_columns


# Fungsi umum untuk perhitungan statistik
def calculate_statistics(df, columns, statistic):
    """
    Menghitung statistik tertentu (median atau mean) untuk kolom-kolom tertentu.
    :param df: DataFrame
    :param columns: Daftar kolom yang ingin dihitung
    :param statistic: 'median' atau 'mean'
    :return: Hasil perhitungan statistik
    """
    valid_columns = validate_columns(df, columns)
    if not valid_columns:
        print("Tidak ada kolom valid untuk dihitung.")
        return None

    try:
        if statistic == 'median':
            return df[valid_columns].median()
        elif statistic == 'mean':
            return df[valid_columns].mean()
        else:
            raise ValueError(f"Statistik '{statistic}' tidak didukung.")
    except Exception as e:
        print(f"Terjadi error saat menghitung statistik: {e}")
        return None


# Membaca file Excel
def load_excel(file_path):
    """
    Membaca file Excel dan mengembalikan dictionary dari semua sheet.
    """
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        return all_sheets
    except FileNotFoundError:
        print(f"Error: File '{file_path}' tidak ditemukan. Pastikan path file benar.")
        exit()
    except Exception as e:
        print(f"Error saat membaca file Excel: {e}")
        exit()


# Main Program
if __name__ == "__main__":
    file_path = 'C:/Users/MRX/Downloads/Data_Puskesmas_Merged.xlsx'
    all_sheets = load_excel(file_path)

    # Proses Perhitungan Pertama (Rata-rata Sheet1)
    sheet_name1 = 'Sheet1'
    if sheet_name1 in all_sheets:
        df1 = all_sheets[sheet_name1]
        columns_to_average1 = ['usia', 'hr', 'glu']
        average_result1 = calculate_statistics(df1, columns_to_average1, 'mean')
        if average_result1 is not None:
            print("\nProses Perhitungan Pertama (Rata-rata Sheet1):")
            print(f"Rata-rata untuk {columns_to_average1}:")
            print(average_result1)
    else:
        print(f"Sheet '{sheet_name1}' tidak ditemukan.")

    # Proses Perhitungan Kedua (Median Sheet2)
    sheet_name2 = 'Sheet1'
    if sheet_name2 in all_sheets:
        df2 = all_sheets[sheet_name2]
        columns_to_median2 = ['chol', 'acd']
        median_result2 = calculate_statistics(df2, columns_to_median2, 'median')
        if median_result2 is not None:
            print("\nProses Perhitungan Kedua (Median Sheet2):")
            print(f"Median untuk {columns_to_median2}:")
            print(median_result2)
    else:
        print(f"Sheet '{sheet_name2}' tidak ditemukan.")
