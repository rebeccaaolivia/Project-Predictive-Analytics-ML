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
  - **Extra Trees Classifier**: Extra Trees (Extremely Randomized Trees) adalah varian dari Random Forest yang membedakan diri melalui pemilihan split point secara acak saat membangun pohon keputusan. Teknik ini memperkenalkan lebih banyak variasi antar pohon dalam ensemble, sehingga dapat meningkatkan kemampuan generalisasi model dan efisiensi waktu pelatihan. Seperti Random Forest, keputusan akhir diambil berdasarkan hasil agregasi dari seluruh pohon [[8](https://www.ejournal.itn.ac.id/index.php/jati/article/view/8797/4781).

## Data Understanding







