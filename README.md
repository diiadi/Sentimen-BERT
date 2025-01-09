### Dokumentasi Menjalankan Aplikasi Streamlit

Berikut adalah langkah-langkah untuk menjalankan aplikasi Streamlit:

1. **Buat Folder Kosong**  
   Buat folder baru di lokasi yang diinginkan untuk menyimpan aplikasi. Misalnya:  
   ```
   mkdir nama_folder
   cd nama_folder
   ```

2. **Ekstrak File ZIP**  
   Ekstrak file ZIP aplikasi ke dalam folder yang telah dibuat.

3. **Buat Virtual Environment**  
   Buat *virtual environment* dengan perintah berikut:  
   ```
   python -m venv env
   ```

4. **Aktifkan Virtual Environment**  
   Aktifkan *virtual environment* menggunakan perintah:  
   - **Windows:**  
     ```
     .\env\Scripts\activate
     ```
   - **Mac/Linux:**  
     ```
     source env/bin/activate
     ```

5. **Install Dependencies**  
   Install semua dependensi yang diperlukan dengan membaca file `requirements.txt`:  
   ```
   pip install -r requirements.txt
   ```

6. **Jalankan Aplikasi**  
   Jalankan aplikasi dengan perintah:  
   ```
   streamlit run app.py
   ```

Aplikasi akan terbuka di *browser* default pada alamat seperti `http://localhost:8501`.