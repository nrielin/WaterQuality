import joblib
import numpy as np
from tkinter import messagebox
import tkinter as tk

column_names = ["aluminium", "ammonia", "arsenic", "barium", "cadmium", "chloramine",
                "chromium", "copper", "flouride", "bacteria", "viruses", "lead",
                "nitrates", "nitrites", "mercury", "perchlorate", "radium", "selenium",
                "silver", "uranium"]

model = joblib.load("Wquality.joblib")

def predict_water_quality(attributes):
    X = np.array(attributes).reshape(1, -1)
    prediction = model.predict(X)
    return "Sạch" if prediction[0] == 1 else "Bẩn"

def on_predict():
    try:
        attributes = [float(entry.get()) for entry in entry_boxes]
        result = predict_water_quality(attributes)
        messagebox.showinfo("Kết quả", f"Chất lượng nước được dự đoán là: {result}")
    except ValueError:
        messagebox.showerror("Lỗi", "Vui lòng nhập các giá trị số cho tất cả các thuộc tính!")
def clear_entries():
    for entry in entry_boxes:
        entry.delete(0, tk.END)

root = tk.Tk()
root.title("Xác định chất lượng nước")


entry_boxes = []
for i, column_name in enumerate(column_names):
    label = tk.Label(root, text=f"Thuộc tính {column_name}:")
    label.grid(row=i, column=0, padx=5, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=5, pady=5)
    entry_boxes.append(entry)


predict_button = tk.Button(root, text="Dự đoán", command=on_predict)
predict_button.grid(row=20, column=0, columnspan=2, pady=10)

clear_button = tk.Button(root, text="Làm mới", command=clear_entries)
clear_button.grid(row=len(column_names), column=1, padx=5, pady=10)

root.mainloop()