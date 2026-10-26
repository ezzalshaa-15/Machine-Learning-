import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)

pred_train = tree_full.predict(X_train)
pred_test = tree_full.predict(X_test)

print("=== DECISION TREE TANPA PRUNING ===")
print("Jumlah node:", tree_full.tree_.node_count)
print("Akurasi training:", round(accuracy_score(y_train, pred_train), 3))
print("Akurasi testing:", round(accuracy_score(y_test, pred_test), 3))

tree_depth = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
tree_depth.fit(X_train, y_train)


tree_leaf = DecisionTreeClassifier(
    min_samples_leaf=5,
    random_state=42
)
tree_leaf.fit(X_train, y_train)

tree_pruned = DecisionTreeClassifier(
    ccp_alpha=0.01,
    random_state=42
)
tree_pruned.fit(X_train, y_train)


models = {
    "Tanpa Pruning": tree_full,
    "Max Depth = 3": tree_depth,
    "Min Samples Leaf = 5": tree_leaf,
    "CCP Alpha = 0.01": tree_pruned
}

print("\n=== PERBANDINGAN MODEL ===")
print(f"{'Model':<25} {'Nodes':<10} {'Train Acc':<12} {'Test Acc':<10}")

for name, model in models.items():
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(
        f"{name:<25} "
        f"{model.tree_.node_count:<10} "
        f"{train_acc:.3f}        "
        f"{test_acc:.3f}"
    )


plt.figure(figsize=(18, 10))

plot_tree(
    tree_pruned,
    feature_names=wine.feature_names,
    class_names=wine.target_names,
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree setelah Pruning")
plt.show()


print("\n=== ATURAN IF-THEN ===")

rules = export_text(
    tree_pruned,
    feature_names=list(wine.feature_names)
)

print(rules)


print("\n=== KESIMPULAN ===")
print(
    "Pruning membatasi kompleksitas pohon keputusan. "
    "Tujuannya mengurangi overfitting agar model lebih mampu "
    "memprediksi data baru."
)
