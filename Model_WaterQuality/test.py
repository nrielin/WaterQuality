# import tkinter as tk
# from tkinter import messagebox
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier

# # Hàm dự đoán chất lượng nước dựa trên 20 thuộc tính đầu vào
# def predict_water_quality(attributes):
#     # Chuẩn bị dữ liệu đầu vào cho mô hình
#     X = np.array(attributes).reshape(1, -1)
#     # Dự đoán nhãn cho mẫu dữ liệu này
#     prediction = model.predict(X)
#     # Trả về kết quả dự đoán (sạch hoặc bẩn)
#     return "Sạch" if prediction[0] == 1 else "Bẩn"

# # Load pre-trained model
# # model = RandomForestClassifier()
# # model.load('your_model_path')  # Load model from file

# def on_predict():
#     try:
#         # Lấy giá trị từ các ô nhập liệu
#         attributes = [float(entry.get()) for entry in entry_boxes]
#         # Dự đoán chất lượng nước
#         result = predict_water_quality(attributes)
#         messagebox.showinfo("Kết quả", f"Chất lượng nước được dự đoán là: {result}")
#     except ValueError:
#         messagebox.showerror("Lỗi", "Vui lòng nhập các giá trị số cho tất cả các thuộc tính!")

# # Tạo cửa sổ chính
# root = tk.Tk()
# root.title("Xác định chất lượng nước")

# # Tạo các ô nhập liệu cho 20 thuộc tính
# entry_boxes = []
# for i in range(20):
#     label = tk.Label(root, text=f"Thuộc tính {i + 1}:")
#     label.grid(row=i, column=0, padx=5, pady=5)
#     entry = tk.Entry(root)
#     entry.grid(row=i, column=1, padx=5, pady=5)
#     entry_boxes.append(entry)

# # Tạo nút "Dự đoán"
# predict_button = tk.Button(root, text="Dự đoán", command=on_predict)
# predict_button.grid(row=20, column=0, columnspan=2, pady=10)

# # Chạy ứng dụng
# root.mainloop()
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

def train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test):
    # Huấn luyện mô hình Naive Bayes
    #model = GaussianNB()
    #model = RandomForestClassifier(n_jobs=-1)
    model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=10)
    model.fit(X_train, y_train)

    # Dự đoán trên tập kiểm tra
    y_pred = model.predict(X_test)

    # Đánh giá mô hình
    accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_mat)

if __name__ == "__main__":
    # Đọc dữ liệu
    data = pd.read_csv('waterQuality1.csv')

    # Tiền xử lý dữ liệu và chia thành tập huấn luyện và tập kiểm tra
    X = data.drop('is_safe', axis=1)
    y = data['is_safe']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện và đánh giá mô hình Naive Bayes
    train_and_evaluate_naive_bayes(X_train, X_test, y_train, y_test)
