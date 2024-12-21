import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
from tkinter import messagebox
import tkinter as tk

df = pd.read_csv('waterQuality1.csv')
encoder = OneHotEncoder(sparse_output=False, drop='first').set_output(transform="pandas")
cat_encoded = encoder.fit_transform(df[['is_safe']])
qp = pd.concat([df,cat_encoded],axis=1)

qp.drop(['is_safe'],axis=1,inplace=True)

x = qp[['aluminium', 'ammonia', 'arsenic', 'barium', 'cadmium', 'chloramine',
       'chromium', 'copper', 'flouride', 'bacteria', 'viruses', 'lead',
       'nitrates', 'nitrites', 'mercury', 'perchlorate', 'radium', 'selenium',
       'silver', 'uranium']]
y= qp['is_safe_1']
print(x.shape)
print(y.shape)
x_test,x_train,y_test,y_train= train_test_split(x,y,test_size=0.2)
model = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=10)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
print(y_pred)
model_score=accuracy_score(y_test,y_pred)
print(model_score)
y_pred2 = model.predict(x_train)
print(y_pred)
accuracy_score(y_train,y_pred2)
print(accuracy_score(y_train,y_pred2))
model.score(x_train,y_train)
model.score(x_test,y_test)
plt.figure(figsize=(12,8))
tree.plot_tree(model.fit(x_train, y_train))
plt.show()

import pandas as pd
from joblib import load

data = pd.read_csv("waterQuality1.csv")

column_names = ["aluminium", "ammonia", "arsenic", "barium", "cadmium", "chloramine",
                "chromium", "copper", "flouride", "bacteria", "viruses", "lead",
                "nitrates", "nitrites", "mercury", "perchlorate", "radium", "selenium",
                "silver", "uranium"]

model_path = "model/Wquality.joblib"
joblib.dump(model, model_path)

print(f"Trying to load model from: {model_path}")
try:
    model = load(model_path)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found. Check the path and existence of the file.")
except PermissionError:
    print("Error: Insufficient permissions to access the model file.")
except UnicodeDecodeError:
    print("Error: Potential encoding mismatch between saved and loaded files.")


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