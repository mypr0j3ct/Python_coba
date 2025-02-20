import pandas as pd

def kalkulasi_dan_buat_excel_baru():
    """
    Membaca file Excel, menghitung RHR, dan membuat file Excel baru
    dengan kolom RHR, age, dan chol.
    """
    try:
        # --- 1. Membaca File Excel Sumber ---
        path_file_lama = r"C:\Users\MRX\Downloads\omaigat.xlsx"  # Gunakan raw string (r"")
        df_lama = pd.read_excel(path_file_lama, sheet_name="Sheet1")

        # --- 2. Periksa Kolom (Penting untuk Validasi) ---
        kolom_dibutuhkan = ["TR", "MR", "age", "chol"]
        for kolom in kolom_dibutuhkan:
            if kolom not in df_lama.columns:
                raise ValueError(f"Kolom '{kolom}' tidak ditemukan di file Excel.")

        # --- 3. Menghitung RHR (Resting Heart Rate) ---
        #  RHR = TR - MR
        df_lama["RHR"] = df_lama["TR"] - df_lama["MR"]

        # --- 4. Memilih Kolom yang Diinginkan ---
        df_baru = df_lama[["RHR", "age", "chol"]].copy()  # .copy() untuk menghindari SettingWithCopyWarning

        # --- 5. Menyimpan ke File Excel Baru ---
        path_file_baru = r"C:\Users\MRX\Videos\Captures\Python_Final\hasil_perhitungan.xlsx"
        df_baru.to_excel(path_file_baru, sheet_name="Sheet1", index=False)  # index=False agar tidak ada kolom index

        print(f"File Excel baru berhasil dibuat di: {path_file_baru}")

    except FileNotFoundError:
        print(f"Error: File Excel tidak ditemukan di path: {path_file_lama}")
    except ValueError as ve:
        print(f"Error: {ve}")
    except Exception as e:
        print(f"Terjadi kesalahan tak terduga: {e}")


# --- Jalankan fungsi utama ---
if __name__ == "__main__":  # Praktik terbaik untuk program yang lebih besar
    kalkulasi_dan_buat_excel_baru()
