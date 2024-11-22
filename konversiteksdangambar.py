import pandas as pd
import numpy as np
import cv2  # Untuk membaca gambar
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Path ke file Excel dan folder gambar
excel_path = "question.xlsx"  # Path file Excel
image_folder = "image"  # Ganti dengan lokasi folder gambar

# Membaca file Excel
data = pd.read_excel(excel_path)

# Pastikan ada 100 soal dalam dataset Excel
print(f"Jumlah soal dalam dataset Excel: {len(data)}")

# Mendapatkan daftar nama file gambar (pastikan gambar di folder sesuai urutan)
image_files = sorted(os.listdir(image_folder))

# Pastikan ada 100 gambar
print(f"Jumlah gambar dalam folder: {len(image_files)}")

# Preprocessing: Mengubah ukuran gambar dan menangani gambar yang gagal dimuat
target_size = (128, 128)  # Ukuran gambar yang diinginkan
images = []
labels = []

for i, row in data.iterrows():
    # Ambil soal dari Excel
    label = row['question']  # Sesuaikan nama kolom ini dengan nama yang sesuai di Excel
    
    # Ambil gambar berdasarkan urutan
    image_path = os.path.join(image_folder, image_files[i])
    
    # Baca gambar, tangani jika gambar gagal dimuat
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Membaca gambar dalam grayscale
    if img is not None:
        # Resize gambar agar seragam
        img_resized = cv2.resize(img, target_size)
        images.append(img_resized)
        labels.append(label)
    else:
        print(f"Gambar gagal dimuat: {image_path}")

# Konversi ke numpy array, pastikan ukuran gambar seragam
images = np.array(images)
labels = np.array(labels)

# Cek jumlah gambar dan label
print(f"Jumlah gambar dan label: {len(images)}")

# Pastikan semua gambar memiliki bentuk yang konsisten
print(f"Shape gambar pertama: {images[0].shape}")
# Membagi dataset (70% train, 20% validation, 10% test)
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)  # 0.33 x 30% = 10%
# Fungsi untuk preprocessing
def preprocess_images(image_array, target_size=(128, 128)):
    processed_images = []
    for img in image_array:
        # Resize gambar
        resized_img = cv2.resize(img, target_size)
        # Normalisasi piksel (0-255 menjadi 0-1)
        normalized_img = resized_img / 255.0
        # Tambahkan dimensi untuk channel (TensorFlow format: (H, W, C))
        processed_images.append(normalized_img[..., np.newaxis])
    return np.array(processed_images)

# Preprocessing
X_train = preprocess_images(X_train)
X_val = preprocess_images(X_val)
X_test = preprocess_images(X_test)

# Cek dimensi
print(f"Shape X_train: {X_train.shape}, y_train: {len(y_train)}")
print(f"Shape X_val: {X_val.shape}, y_val: {len(y_val)}")
print(f"Shape X_test: {X_test.shape}, y_test: {len(y_test)}")
#Buat dictionary karakter
all_text = ''.join(labels)  # Gabungkan semua teks label
characters = sorted(set(all_text))  # Semua karakter unik
char_to_num = {char: idx + 1 for idx, char in enumerate(characters)}  # Mulai dari 1 (0 untuk padding)
num_to_char = {idx: char for char, idx in char_to_num.items()}

# Fungsi encoding
def encode_labels(labels, max_length):
    encoded = []
    for label in labels:
        # Encode setiap karakter dalam label
        encoded_label = [char_to_num[char] for char in label]
        # Padding jika lebih pendek dari max_length
        if len(encoded_label) < max_length:
            encoded_label += [0] * (max_length - len(encoded_label))
        encoded.append(encoded_label[:max_length])  # Potong jika terlalu panjang
    return np.array(encoded)

# Panjang maksimum label
max_label_length = max([len(label) for label in labels])

# Encode labels
y_train_encoded = encode_labels(y_train, max_label_length)
y_val_encoded = encode_labels(y_val, max_label_length)
y_test_encoded = encode_labels(y_test, max_label_length)

print(f"Sample Encoded Label: {y_train_encoded[0]}")