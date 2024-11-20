import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load data
train_data = pd.read_csv('./DataMining/input/train.csv')
test_data = pd.read_csv('./DataMining/input/test.csv')

print("Train Data Info:")
print(train_data.info())
print("Train Data Head:")
print(train_data.head())

print("Test Data Info:")
print(test_data.info())
print("Test Data Head:")
print(test_data.head())

# Kiểm tra và loại bỏ các bản ghi trùng lặp
duplicate_train = train_data.duplicated()
print("Number of duplicate train rows:", duplicate_train.sum())
duplicate_test = test_data.duplicated()
print("Number of duplicate test rows:", duplicate_test.sum())

train_data = train_data.drop_duplicates()
test_data = test_data.drop_duplicates()

# Kiểm tra và xử lý các giá trị thiếu
missing_train = train_data.isnull().sum()
missing_test = test_data.isnull().sum()
print("Missing values in train data:\n", missing_train)
print("Missing values in test data:\n", missing_test)

# Loại bỏ các hàng chứa giá trị thiếu
train_data = train_data.dropna()
test_data = test_data.dropna()

# Hoặc thay thế các giá trị thiếu bằng giá trị trung bình (hoặc giá trị khác)
# train_data = train_data.fillna(train_data.mean())
# test_data = test_data.fillna(test_data.mean())

# Mã hóa các giá trị chuỗi
label_encoders = {}
for column in train_data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    train_data[column] = le.fit_transform(train_data[column])
    test_data[column] = le.transform(test_data[column])
    label_encoders[column] = le

# Giả sử 'satisfaction' là cột mục tiêu và các cột còn lại là đặc trưng
X_train_full = train_data.drop('satisfaction', axis=1)
y_train_full = train_data['satisfaction']
X_test = test_data.drop('satisfaction', axis=1)
y_test = test_data['satisfaction']

# Chia thành tập huấn luyện và kiểm tra nội bộ
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Huấn luyện mô hình SVM
svm_model = SVC(kernel='linear', random_state=42, probability=True)
svm_model.fit(X_train, y_train)

# Dự đoán
y_val_pred = svm_model.predict(X_val)
y_test_pred = svm_model.predict(X_test)

# Đánh giá
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("Validation Report:\n", classification_report(y_val, y_val_pred))
print("Test Report:\n", classification_report(y_test, y_test_pred))

# Ma trận nhầm lẫn
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
conf_matrix_test = confusion_matrix(y_test, y_test_pred)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues')
plt.title('Validation Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.subplot(1, 2, 2)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

plt.show()

# Biểu đồ ROC và AUC
y_val_prob = svm_model.predict_proba(X_val)[:, 1]
y_test_prob = svm_model.predict_proba(X_test)[:, 1]

fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob)
roc_auc_val = auc(fpr_val, tpr_val)

fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
roc_auc_test = auc(fpr_test, tpr_test)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr_val, tpr_val, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Validation ROC Curve')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(fpr_test, tpr_test, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curve')
plt.legend(loc="lower right")

plt.show()