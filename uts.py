import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import numpy as np

# Menambahkan label kategori kebugaran berdasarkan Resting Heart Rate Before
def categorize_fitness(rhr):
    if rhr < 60:
        return "Sangat Baik"
    elif 60 <= rhr < 75:
        return "Baik"
    elif 75 <= rhr < 90:
        return "Cukup"
    else:
        return "Kurang Baik"

# Membaca dataset
data = pd.read_csv("C:\\Users\\asust\\OneDrive\\Documents\\UTS KB\\dataset\\heart_rate_data.csv")  # Ganti dengan path dataset Anda
data['Fitness Category'] = data['Resting Heart Rate Before'].apply(categorize_fitness)

# Plot distribusi kategori fitness
plt.figure(figsize=(8, 6))
sns.countplot(x='Fitness Category', data=data, palette='viridis')
plt.title('Distribusi Kategori Fitness')
plt.xlabel('Kategori Fitness')
plt.ylabel('Jumlah')
plt.show()

# Memisahkan fitur dan label
X = data.drop(columns=['Fitness Category'])  # Semua kolom selain label
y = data['Fitness Category']

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing: Standarisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model menggunakan SVM
model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)  # SVM dengan kernel RBF
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=model.classes_)
disp.plot(cmap='viridis', xticks_rotation='vertical')
plt.title('Confusion Matrix')
plt.show()

# Menampilkan hasil evaluasi
print("\nEvaluasi Model:")
print("Confusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Prediksi contoh baru
new_data = [[30, 7.5, 5, 70, 65, 150]]  # Contoh data baru

# Buat DataFrame untuk data baru dengan nama kolom yang sesuai
new_data_df = pd.DataFrame(new_data, columns=X.columns)

# Standardisasi data baru
new_data_scaled = scaler.transform(new_data_df)

# Prediksi
prediction = model.predict(new_data_scaled)
print("\nPrediction for new data (Fitness Category):", prediction[0])
