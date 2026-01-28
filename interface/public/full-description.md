# Panduan Pengguna Agen Identifikasi Penyakit Tanaman

Selamat datang di antarmuka Agen Identifikasi Penyakit Tanaman. Panduan ini dirancang untuk membantu Anda memanfaatkan fitur-fitur sistem secara optimal dalam mendiagnosis dan memantau kesehatan tanaman.

## 🚀 Memulai

### Cara Memulai Percakapan

**Input Langsung**

Ketik pertanyaan atau keluhan mengenai tanaman Anda pada kolom input di bagian bawah layar, lalu tekan tombol Kirim atau tekan Enter.

**Contoh Prompt**

Gunakan tombol contoh pertanyaan yang muncul di awal percakapan untuk memulai interaksi dengan cepat.

### ⌨️ Pintasan Keyboard

- `Enter`: Mengirim pesan
- `Shift + Enter`: Menambahkan baris baru
- `Cmd/Ctrl + Enter`: Mengirim pesan dalam mode edit

## ✨ Fitur Utama

### 📎 Unggah File

Anda dapat melampirkan foto tanaman atau dokumen PDF untuk dianalisis oleh agen.

**Cara Mengunggah**

- Klik ikon klip kertas di sebelah kiri kolom input untuk memilih file
- Tarik dan lepas (drag and drop) file ke area obrolan
- Salin dan tempel (paste) gambar dari clipboard

**Format yang Didukung**

- Gambar: JPEG, PNG, GIF, WebP (Disarankan untuk foto gejala penyakit tanaman)
- Dokumen: PDF

### 🔧 Kontrol Tampilan Alat (Tools)

Agen ini menggunakan berbagai alat canggih seperti **Deteksi Objek** (untuk melokalisasi gejala pada daun) dan **Pencarian Multimodal** (untuk mencocokkan gejala dengan basis data penyakit). Klik ikon kunci inggris (wrench) untuk menampilkan atau menyembunyikan detail proses penggunaan alat tersebut.

**Mode Tampil**: Anda dapat melihat langkah-langkah berpikir agen, termasuk alat apa yang digunakan (misalnya: memindai gambar, mencari referensi).
**Mode Sembunyi**: Hanya hasil akhir dan jawaban agen yang ditampilkan agar lebih ringkas.

### 📚 Manajemen Riwayat Percakapan

**Akses Sidebar**

Klik tombol toggle di pojok kiri atas untuk membuka sidebar riwayat. Anda dapat melihat dan memilih percakapan sebelumnya dari daftar ini.

**Judul Percakapan**

Setiap sesi percakapan akan diberi judul otomatis. Anda dapat mengedit judul ini agar lebih mudah ditemukan kembali.

**Menghapus Percakapan**

Percakapan yang tidak lagi diperlukan dapat dihapus melalui opsi hapus pada masing-masing item di daftar.

### 🔄 Regenerasi Respons

Jika jawaban agen dirasa kurang memuaskan, klik tombol regenerasi di bawah pesan terakhir untuk mendapatkan respons baru. Agen akan mencoba memberikan jawaban dengan sudut pandang atau penjelasan yang berbeda.

## ⚙️ Pengaturan

Akses menu pengaturan melalui tombol di pojok kanan bawah layar.

### 🎨 Tampilan

**Tema Warna**

- Mode Terang: Tampilan dengan latar belakang cerah
- Mode Gelap: Tampilan dengan latar belakang gelap yang nyaman di mata
- Mode Otomatis: Mengikuti pengaturan sistem perangkat Anda

**Gaya Font**

- Sans Serif: Font standar yang bersih
- Serif: Font dengan kait yang nyaman dibaca
- Monospace: Font lebar tetap, cocok untuk menampilkan kode

**Ukuran Teks**

Pilih antara Kecil, Sedang, atau Besar sesuai kenyamanan baca Anda.

### 💡 Perilaku Antarmuka

**Lebar Obrolan**

- Standar: Lebar menengah (768px) untuk fokus membaca
- Lebar: Tampilan lebih luas (1280px), cocok untuk melihat gambar hasil deteksi yang besar

**Lipat Alat Otomatis**

Jika diaktifkan, detail penggunaan alat akan otomatis disembunyikan setelah agen selesai menjawab, menjaga tampilan tetap rapi.

## 🎯 Penggunaan Lanjutan

### Konteks Berkelanjutan

Agen mengingat konteks percakapan sebelumnya. Anda tidak perlu mengulang informasi yang sudah disampaikan dalam satu sesi.

**Contoh**

```
Pengguna: [Mengunggah foto daun tomat] "Apa penyakit pada tanaman ini?"
AI: "Berdasarkan analisis visual, ini terlihat seperti hawar daun awal (Early Blight) yang disebabkan oleh jamur Alternaria solani."
Pengguna: "Bagaimana cara mengobatinya secara organik?"
AI: [Memberikan rekomendasi penanganan organik untuk hawar daun]
```

### 📝 Dukungan Markdown

Jawaban agen diformat menggunakan Markdown untuk keterbacaan yang lebih baik:

- **Judul**: Struktur hierarkis informasi
- **Daftar**: Poin-poin atau langkah-langkah
- **Tautan**: Referensi ke sumber eksternal
- **Blok Kode**: Menampilkan kode atau data terstruktur
- **Tabel**: Menyajikan data perbandingan

### 💻 Kode & Data

Jika agen memberikan output berupa kode atau data JSON, sintaksisnya akan diwarnai otomatis. Anda dapat menyalinnya dengan mudah menggunakan tombol salin di pojok blok kode.

## 🔍 Pemecahan Masalah

### Pesan Tidak Terkirim

1. Periksa koneksi internet Anda.
2. Segarkan (refresh) halaman browser.
3. Jika masalah berlanjut, cek konsol pengembang (developer tools) pada browser.

### Gagal Mengunggah File

1. Pastikan ukuran file tidak melebihi batas.
2. Pastikan format file didukung (JPEG, PNG, GIF, WebP, PDF).
3. Pastikan browser memiliki izin akses file.

### Riwayat Tidak Muncul

1. Buka sidebar menggunakan tombol di pojok kiri atas.
2. Cek pengaturan apakah opsi riwayat diaktifkan.

### Pengaturan Tidak Berubah

Lakukan *hard refresh* pada browser (Ctrl+Shift+R atau Cmd+Shift+R) untuk membersihkan cache.

## 💡 Tips Bermanfaat

### Foto yang Jelas

Untuk hasil identifikasi penyakit yang akurat, pastikan foto yang diunggah memiliki pencahayaan yang baik, fokus (tidak buram), dan menampilkan gejala penyakit dengan jelas.

### Pertanyaan Spesifik

Berikan informasi tambahan seperti jenis tanaman atau berapa lama gejala sudah muncul untuk membantu agen memberikan diagnosis yang lebih presisi.

### Pisahkan Topik

Gunakan percakapan baru untuk topik atau tanaman yang berbeda agar konteks analisis tidak tercampur.

## 🖥️ Spesifikasi Teknis

### Browser yang Didukung

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

### Lingkungan yang Disarankan

- Koneksi internet stabil (diperlukan untuk akses model AI dan basis data)
- Browser modern dengan JavaScript aktif

## 📚 Informasi Tambahan

Untuk dokumentasi teknis lebih lanjut mengenai metode DDR, arsitektur sistem, atau implementasi model YOLO dan SCOLD, silakan merujuk pada dokumen Skripsi atau repositori proyek.

- Repositori: [GitHub Link](https://github.com/andyathsid/thesis)
- Dokumentasi LangChain: [LangChain Documentation](https://docs.langchain.com/)
- Dokumentasi Next.js: [nextjs.org/docs](https://nextjs.org/docs)

---

