import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

# ==============================
# Hàm train model và xuất luật
# ==============================
def train_and_extract_rules(data):
    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train model với entropy
    clf = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf.fit(X_train, y_train)

    # Accuracy
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Xuất cây
    tree_rules = export_text(clf, feature_names=list(X.columns))

    # Rút luật IF – THEN
    rules = []
    for line in tree_rules.split("\n"):
        if "class:" in line:
            rules.append(line.strip())

    return acc, tree_rules, rules


# ==============================
# Streamlit UI
# ==============================
st.title("🌳 Phân lớp bệnh nhân tiểu đường bằng Cây quyết định")

uploaded_file = st.file_uploader("📂 Upload file CSV (có cột Outcome)", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Dữ liệu ban đầu")
    st.write(data.head())

    acc, tree_rules, rules = train_and_extract_rules(data)

    st.success(f"🎯 Accuracy: {acc*100:.2f}%")

    st.subheader("🌳 Cây quyết định")
    st.text(tree_rules)

    st.subheader("📜 Luật IF – THEN")
    for r in rules:
        st.text(r)
