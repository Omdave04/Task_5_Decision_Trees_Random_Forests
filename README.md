import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

df = pd.read_csv('/mnt/data/heart.csv')
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Classifier")
plt.show()

print(f"Train Accuracy (DT): {dt.score(X_train, y_train):.2f}")
print(f"Test Accuracy (DT): {dt.score(X_test, y_test):.2f}")

dt_limited = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_limited.fit(X_train, y_train)
print("Test Accuracy (DT with max_depth=4):", dt_limited.score(X_test, y_test))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
print("Test Accuracy (Random Forest):", accuracy_score(y_test, rf_preds))

importances = rf.feature_importances_
feat_importances = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
sns.barplot(x=feat_importances, y=feat_importances.index)
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

cv_scores_dt = cross_val_score(dt_limited, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
print(f"Cross-validation Accuracy (Decision Tree, depth=4): {cv_scores_dt.mean():.2f}")
print(f"Cross-validation Accuracy (Random Forest): {cv_scores_rf.mean():.2f}")
