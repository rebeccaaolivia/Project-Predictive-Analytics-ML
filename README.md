# Laporan Proyek Machine Learning - Rebecca Olivia

## Domain Proyek
Domain yang dipilih untuk proyek _machine learning_ ini adalah **Kesehatan**, dengan judul **Predictive Analytics: Prediksi Kanker**.

### Latar Belakang
![dataset-cover](https://github.com/user-attachments/assets/609efb12-0a90-4358-9839-70f56b915838)

Beban penyakit kanker di Indonesia tergolong tinggi. Mengacu pada data _Global Burden of Cancer Study_ (GLOBOCAN) tahun 2020, dari total 19,3 juta kasus kanker secara global, Indonesia mencatatkan 396.914 kasus dan 234.511 kematian yang disebabkan oleh kanker [[1](https://ejournal.unib.ac.id/JurnalVokasiKeperawatan/article/view/22338/10237)]. Lima jenis kanker yang paling banyak ditemui baik pada laki-laki maupun perempuan Indonesia adalah kanker payudara, paru, serviks, kolorektal atau usus besar dan rektum, serta hati [[2](https://ugm.ac.id/id/berita/jumlah-penderita-kanker-terus-meningkat-kenali-gejala-awal-untuk-deteksi-dini/#:~:text=Dari%20sejumlah%20kasus%20yang%20ada,besar%20dan%20rektum%2C%20serta%20hati.)].

Salah satu tantangan terbesar dalam penanganan kanker adalah keterlambatan dalam proses diagnosis. Banyak pasien baru teridentifikasi mengidap kanker ketika penyakit sudah memasuki stadium lanjut, sehingga peluang kesembuhan menjadi sangat kecil. Untuk itu, deteksi dini memegang peranan penting dalam meningkatkan angka harapan hidup. Upaya ini dapat dilakukan melalui pemeriksaan kesehatan secara rutin serta pengenalan terhadap gejala awal. Selain itu, prediksi yang akurat juga menjadi kunci untuk memperbesar kemungkinan sembuh dan mengurangi angka kematian akibat kanker [[3](https://www.ejournal.itn.ac.id/index.php/jati/article/view/10752/6190)].

Penerapan *predictive analytics* dalam bidang kesehatan dapat membantu mendeteksi risiko kanker sejak dini menggunakan data rekam medis dan gaya hidup. Model prediksi ini dapat membantu:

- **Tenaga medis**: untuk menyaring pasien yang berisiko tinggi secara cepat dan efisien.
- **Pasien**: agar bisa mengambil langkah preventif lebih awal.
- **Pemerintah dan institusi kesehatan**: dalam menyusun kebijakan berbasis data dan efisiensi sumber daya.

Dalam proyek ini, digunakan pendekatan machine learning berbasis klasifikasi untuk memprediksi apakah seseorang berpotensi mengidap kanker atau tidak, berdasarkan fitur seperti usia, jenis kelamin, indeks massa tubuh (IMT), kebiasaan merokok, risiko genetik, tingkat aktivitas fisik, konsumsi alkohol mingguan, riwayat pribadi terhadap kanker, serta status diagnosis.

## Business Understanding
Pengembangan model prediksi kanker sangat penting dalam mendukung diagnosis dini dan penanganan penyakit secara lebih efektif. Dengan memanfaatkan data medis seperti data visual dan sensorik (misalnya ukuran benjolan, tekstur, atau hasil pemeriksaan laboratorium), model ini dapat membantu para tenaga medis dalam mengidentifikasi potensi kanker sejak dini. Selain itu, model ini juga dapat meningkatkan efisiensi layanan kesehatan dan mengurangi beban kerja dokter dalam proses diagnosis awal. Prediksi yang akurat akan berdampak besar terhadap kualitas hidup pasien melalui tindakan penanganan yang lebih cepat dan tepat sasaran.

### Problem Statements
Berdasarkan latar belakang di atas, maka permasalahan yang akan dijawab dalam proyek ini adalah:
1. Bagaimana membangun model machine learning yang dapat memprediksi kemungkinan seseorang mengidap kanker berdasarkan data fitur medis?
2. Model machine learning apa yang memiliki akurasi terbaik dalam memprediksi kasus kanker?
3. Bagaimana penerapan model ini dapat membantu meningkatkan efektivitas diagnosis dini dalam sistem layanan kesehatan?

### Goals
Proyek ini memiliki tujuan sebagai berikut:
1. Mengembangkan model machine learning untuk memprediksi kemungkinan seseorang menderita kanker berdasarkan data medis (fitur numerik dan kategorikal).
2. Membandingkan beberapa algoritma klasifikasi untuk menentukan model dengan performa terbaik.
3. Menyediakan solusi yang dapat mendukung diagnosis awal dan pengambilan keputusan medis yang lebih cepat dan akurat.

### Solution Statements
Untuk mencapai tujuan proyek ini, langkah-langkah yang dilakukan meliputi:

- **Eksplorasi dan Analisis Data**  
  Melakukan analisis univariat dan multivariat terhadap fitur-fitur dalam dataset untuk memahami distribusi data, mengidentifikasi korelasi antar fitur, serta mendeteksi outlier yang berpotensi memengaruhi kinerja model.

- **Preprocessing Data**  
  Melakukan pembersihan data (handling missing values dan duplikasi) dan normalisasi agar data siap digunakan dalam proses pelatihan model machine learning sehingga dapat menghasilkan prediksi yang optimal.

- **Pembangunan dan Evaluasi Model**  
  Membangun dan membandingkan beberapa algoritma machine learning untuk menentukan model terbaik dalam melakukan prediksi. Model yang digunakan antara lain:
  
  - **K-Nearest Neighbors (KNN)**: KNN merupakan algoritma klasifikasi yang bekerja dengan prinsip kesamaan jarak. Dalam proses prediksinya, algoritma ini akan mencari sejumlah _k_ tetangga terdekat dari data uji berdasarkan jarak tertentu, seperti Euclidean distance. Objek baru kemudian diklasifikasikan ke dalam kelas mayoritas dari tetangga terdekat tersebut. Pendekatan ini bersifat non-parametrik dan cocok digunakan ketika hubungan antar fitur bersifat non-linear [[4](https://journal.irpi.or.id/index.php/malcom/article/view/1078/519)].
  - **Random Forest**: Random Forest adalah algoritma ensemble yang terdiri dari banyak pohon keputusan (_decision trees_) yang dibangun dari subset acak data dan fitur. Setiap pohon membuat prediksi secara mandiri, kemudian hasil akhirnya ditentukan melalui voting mayoritas. Pendekatan ini meningkatkan akurasi dan kestabilan model serta mampu mengurangi risiko overfitting yang kerap terjadi pada model _decision tree_ tunggal [[5](https://jidt.org/jidt/article/view/393/205)].
  - **Support Vector Machine (SVM)**: SVM merupakan algoritma supervised learning yang berfungsi untuk memisahkan data ke dalam dua kelas atau lebih dengan mencari hyperplane terbaik yang memaksimalkan margin antar kelas. Metode ini dapat bekerja pada ruang data berdimensi tinggi dan juga mampu menangani klasifikasi non-linear menggunakan kernel trick, sehingga sangat andal untuk berbagai jenis permasalahan klasifikasi [[6](https://ejournal.poltekharber.ac.id/index.php/informatika/article/view/977/795)].
  - **Naive Bayes**: Naive Bayes adalah metode klasifikasi berbasis probabilitas yang menerapkan Teorema Bayes dengan asumsi independensi antar fitur. Meskipun asumsi ini jarang terpenuhi secara ketat dalam data nyata, algoritma ini terbukti efisien dan cukup akurat dalam berbagai kasus. Kelebihan utama Naive Bayes adalah kesederhanaannya dalam implementasi dan kecepatan dalam melakukan prediksi [[7](https://ojs.stikombanyuwangi.ac.id/index.php/jikom/article/view/280/147)].
  - **Extra Trees Classifier**: Extra Trees (Extremely Randomized Trees) adalah varian dari Random Forest yang membedakan diri melalui pemilihan split point secara acak saat membangun pohon keputusan. Teknik ini memperkenalkan lebih banyak variasi antar pohon dalam ensemble, sehingga dapat meningkatkan kemampuan generalisasi model dan efisiensi waktu pelatihan. Seperti Random Forest, keputusan akhir diambil berdasarkan hasil agregasi dari seluruh pohon [[8](https://www.ejournal.itn.ac.id/index.php/jati/article/view/8797/4781)].

## Data Understanding
### Gathering Data
Dataset yang digunakan dalam proyek ini adalah "The Cancer Data V2", yang berisi informasi mengenai karakteristik pasien serta status diagnosis kanker. File dataset diunggah dari penyimpanan lokal ke lingkungan kerja Google Colab dalam format ZIP, kemudian diekstrak dan dibaca menggunakan library pandas.

Proses pengumpulan data dilakukan melalui tiga langkah utama:
1. Mengunggah file ZIP (The_Cancer_data_1500_V2.csv.zip) dari komputer lokal ke Google Colab.
2. Mengekstrak isi file ZIP untuk mendapatkan file CSV.
3. Membaca file CSV ke dalam bentuk DataFrame agar dapat dianalisis lebih lanjut.

Dataset ini berisi 1.500 entri dan 9 fitur yang merepresentasikan karakteristik pasien, seperti usia, jenis kelamin, indeks massa tubuh (IMT), kebiasaan merokok, risiko genetik, tingkat aktivitas fisik, konsumsi alkohol mingguan, riwayat pribadi terhadap kanker, serta status diagnosis.

### Data Assesing and Data Cleaning
Informasi Datasets

| Jenis       | Keterangan                                                                                                                                      |
|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| **Title**   | 🩺📊 Cancer Prediction Dataset 🌟🔬                                                                                                               |
| **Source**  | [Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset)                                                              |
| **Maintainer** | [Rabie El Kharoua](https://www.kaggle.com/rabieelkharoua)                                                                                     |
| **License** | Attribution 4.0 International                                                                                                                   |
| **Visibility** | Publik                                                                                                                                       |
| **Tags**    | Earth and Nature, Cancer, Tabular, Health Conditions, Binary Classification                                                                     |
| **Usability** | 10.00                                                                                                                                         |

> Klik pada nama **Kaggle** untuk langsung menuju ke dataset.
> Klik pada nama **Rabie El Kharoua** untuk langsung menuju ke user pemilik datasey

**Dataset Info**
- Format: CSV (Comma-Separated Values)
- Jumlah data: **1.500 baris** dan **9 kolom**
- Semua kolom memiliki **1500 nilai non-null** (tidak ada missing values)
- Tipe data:
  - `int64` (6 kolom): untuk data diskrit (contoh: usia, biner)
  - `float64` (3 kolom): untuk data kontinu (contoh: BMI, aktivitas fisik)

Contoh data:
| Age | Gender | BMI       | Smoking | GeneticRisk | PhysicalActivity | AlcoholIntake | CancerHistory | Diagnosis |
|-----|--------|-----------|---------|--------------|------------------|----------------|----------------|------------|
| 58  | 1      | 16.08     | 0       | 1            | 8.15             | 4.15           | 1              | 1          |
| 71  | 0      | 30.83     | 0       | 1            | 9.36             | 3.52           | 0              | 0          |
| 48  | 1      | 38.78     | 0       | 2            | 5.14             | 4.73           | 0              | 1          |

---

**Feature Explanation**

| Feature Name       | Tipe Data | Deskripsi                                                                                  |
|--------------------|-----------|---------------------------------------------------------------------------------------------|
| `Age`              | Integer   | Usia pasien dalam tahun. Rentang antara 20 hingga 80 tahun.                                |
| `Gender`           | Integer   | Jenis kelamin pasien. `0` = laki-laki, `1` = perempuan.                                     |
| `BMI`              | Float     | Indeks massa tubuh pasien. Nilai berkisar antara 15 – 40.                                   |
| `Smoking`          | Integer   | Status merokok pasien. `0` = tidak merokok, `1` = merokok.                                  |
| `GeneticRisk`      | Integer   | Risiko genetik terhadap kanker. `0` = rendah, `1` = sedang, `2` = tinggi.                   |
| `PhysicalActivity` | Float     | Jam aktivitas fisik per minggu. Rentang 0 – 10 jam/minggu.                                 |
| `AlcoholIntake`    | Float     | Konsumsi alkohol per minggu. Rentang 0 – 5 satuan/minggu.                                   |
| `CancerHistory`    | Integer   | Riwayat pribadi terhadap kanker. `0` = tidak ada, `1` = ada.                                |
| `Diagnosis`        | Integer   | Label target diagnosis. `0` = tidak terdiagnosis kanker, `1` = terdiagnosis kanker.         |

### Exploratory Data Analysis (EDA)
Tahap *Exploratory Data Analysis (EDA)* dilakukan untuk memahami karakteristik data secara menyeluruh sebelum memasuki proses *data preprocessing* dan pemodelan. Proses ini bertujuan untuk:

- Mengetahui distribusi data dan proporsi target.
- Mengidentifikasi hubungan antar fitur.
- Mendeteksi nilai pencilan (*outliers*).
- Mengatasi ketidakseimbangan kelas pada label target.

---
1. Univariate Analysis  
<p align="center">
  <img src="https://github.com/user-attachments/assets/4383e96a-86e6-4101-9f3d-0f72583c02aa" alt="Gambar 1. Pie-Chart Cancer & Non-Cancer" width="500"/>
</p>

<p align="center"><strong>Gambar 1.</strong> Pie-Chart Kanker & Kanker</p>

Pie chart di atas menunjukkan proporsi pasien yang terdiagnosis kanker (`1`) dan non-kanker (`0`). Sebanyak <strong>557 pasien (37.1%)</strong> terdiagnosis kanker (warna pink), sedangkan <strong>943 pasien (62.9%)</strong> tidak terdiagnosis kanker (warna biru).  Informasi ini penting untuk mengetahui apakah kelas target seimbang atau tidak, karena ketidakseimbangan dapat menyebabkan bias dalam model klasifikasi.


<p align="center">
  <img src="https://github.com/user-attachments/assets/5cd2cb5e-f1c3-4049-81e8-afe16b3d869a" alt="Gambar 2. Histogram Fitur Numerik" width="500"/>
</p>

<p align="center"><strong>Gambar 2.</strong> Histogram Fitur Numerik</p>

Histogram digunakan untuk melihat distribusi dari seluruh fitur numerik. Sebagian besar fitur menunjukkan distribusi yang tidak simetris, dengan beberapa fitur memiliki penyebaran yang sempit dan lainnya menunjukkan kemungkinan adanya outlier. Analisis ini membantu menentukan perlunya transformasi data atau penanganan nilai ekstrim.

---
2. Multivariate Analysis
<p align="center">
  <img src="https://github.com/user-attachments/assets/0710c837-434c-44a0-af84-598276949b4a" alt="Gambar 3. Pairplot Diagnosis terhadap Fitur" width="500"/>
</p>

<p align="center"><strong>Gambar 3.</strong> Pairplot Diagnosis terhadap Fitur</p>

Pairplot memberikan gambaran hubungan antar fitur berdasarkan label diagnosis. Warna berbeda menunjukkan kategori kanker dan non-kanker. Beberapa fitur menunjukkan pemisahan yang cukup jelas antar kelas, yang menjadi indikasi baik bahwa fitur-fitur ini berpotensi kuat untuk pemodelan prediktif.


<p align="center">
  <img src="https://github.com/user-attachments/assets/6dfe8b5c-192b-47e0-9550-6ec230cc42eb" alt="Gambar 4. Heatmap Korelasi Antar Fitur" width="500"/>
</p>

<p align="center"><strong>Gambar 4.</strong> Heatmap Korelasi Antar Fitur</p>

Warna merah menunjukkan korelasi positif tinggi, sedangkan biru menunjukkan korelasi negatif. Terlihat bahwa fitur-fitur seperti `radius_mean`, `perimeter_mean`, dan `area_mean` sangat berkorelasi satu sama lain. Informasi ini berguna dalam pemilihan fitur atau teknik reduksi dimensi.

---
3. Outlier Detection and Handling
<p align="center">
  <img src="https://github.com/user-attachments/assets/39281b8e-2fa5-44df-858d-814c56551339" alt="Gambar 5. Boxplot Sebelum Penanganan Outlier" width="500"/>
</p>

<p align="center"><strong>Gambar 5.</strong> Boxplot Sebelum Penanganan Outlier</p>

Boxplot menunjukkan nilai ekstrim (outlier) pada sebagian besar fitur. Outlier dapat mempengaruhi performa model dan perlu ditangani secara hati-hati.

Setelah proses deteksi dan pembersihan menggunakan metode Interquartile Range (IQR), data menjadi lebih bersih:

<p align="center">
  <img src="https://github.com/user-attachments/assets/65f9ea0c-ebcf-41f7-afd0-5c406a09e5b3" alt="Gambar 6. Boxplot Setelah Penangan Outliers" width="500"/>
</p>

<p align="center"><strong>Gambar 6.</strong> Boxplot Setelah Penanganan Outlier</p>

Fitur-fitur setelah penanganan menunjukkan distribusi yang lebih stabil tanpa banyak nilai pencilan ekstrem.

---
4. Class Imbalance & SMOTE
**Distribusi Kelas Sebelum Oversampling**

Distribusi awal kelas target:
- `Kelas 0` (non-kanker): 907 data
- `Kelas 1` (kanker): 377 data
Distribusi ini menunjukkan ketimpangan kelas (class imbalance) yang signifikan. Jika tidak ditangani, model akan cenderung bias terhadap kelas mayoritas.

**Output Distribusi Kelas Setelah SMOTE**

Distribusi kelas setelah oversampling: Counter({0: 907, 1: 907})

Teknik SMOTE (Synthetic Minority Oversampling Technique) digunakan untuk menyeimbangkan kelas dengan menambahkan data sintetis pada kelas minoritas (kanker).

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ce69ebf-c968-4f30-98e4-653ba124b23c" alt="Gambar 7. Visualisasi Sebelum dan Sesudah SMOTE" width="500"/>
</p>

<p align="center"><strong>Gambar 7.</strong> Visualisasi Sebelum dan Sesudah SMOTE</p>

Gambar ini membandingkan jumlah data antar kelas:
- Sebelum SMOTE: kelas kanker (merah) jauh lebih sedikit dibanding non-kanker (hijau),
- Setelah SMOTE: distribusi menjadi seimbang (907 data per kelas), sehingga model dapat dilatih dengan adil dan tidak bias.

## Data Preparation


## Modeling


## Evaluation


## Referensi





