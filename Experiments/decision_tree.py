from pathlib import Path

import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def evaluate_tree(X_train, X_test, y_train, y_test, name):
    model = DecisionTreeClassifier(random_state=42, min_samples_leaf=2)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "predictions": predictions,
        "probabilities": probabilities
    }

    return results


def run_decision_tree_experiment(X_train, X_test, X_train_swd, X_test_swd,
                                  X_train_geometry, X_test_geometry, y_train, y_test, results_file):

    print("\n")
    print("DECISION TREE CLASSIFICATION")
    print("=" * 100)

    original = evaluate_tree(X_train, X_test, y_train, y_test, "Original")
    swd = evaluate_tree(X_train_swd, X_test_swd, y_train, y_test, "SWD-only")
    geometry = evaluate_tree(X_train_geometry, X_test_geometry, y_train, y_test, "SWD + Geometry")

    results = {
        "tree_original": original,
        "tree_swd": swd,
        "tree_geometry": geometry
    }

    print(f"\n{'Representation':<25}{'Accuracy':>12}{'Balanced':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'ROC-AUC':>12}")
    print("-" * 100)

    for name, result in [("Original", original), ("SWD-only", swd), ("SWD + Geometry", geometry)]:
        print(
            f"{name:<25}"
            f"{result['accuracy']:>12.4f}"
            f"{result['balanced_accuracy']:>12.4f}"
            f"{result['precision']:>12.4f}"
            f"{result['recall']:>12.4f}"
            f"{result['f1']:>12.4f}"
            f"{result['roc_auc']:>12.4f}"
        )

    print("\n" + "=" * 80)
    print("IMPROVEMENT OVER ORIGINAL")
    print("=" * 80)

    for name, result in [("SWD-only", swd), ("SWD + Geometry", geometry)]:
        print(f"\n{name}")
        print(f"Accuracy  : {result['accuracy'] - original['accuracy']:+.4f}")
        print(f"Balanced  : {result['balanced_accuracy'] - original['balanced_accuracy']:+.4f}")
        print(f"Precision : {result['precision'] - original['precision']:+.4f}")
        print(f"Recall    : {result['recall'] - original['recall']:+.4f}")
        print(f"F1        : {result['f1'] - original['f1']:+.4f}")
        print(f"ROC-AUC   : {result['roc_auc'] - original['roc_auc']:+.4f}")

    # Detailed classification reports
    for name, result in [("Original", original), ("SWD-only", swd), ("SWD + Geometry", geometry)]:
        print("\n" + "=" * 70)
        print(name)
        print("=" * 70)
        print(classification_report(y_test, result["predictions"]))

    # Update experiment results
    results_path = Path(results_file)
    existing = torch.load(results_path, weights_only=False)

    for name, result in [
        ("tree_original", original),
        ("tree_swd", swd),
        ("tree_geometry", geometry)
    ]:
        existing[name] = {
            "accuracy": result["accuracy"],
            "balanced_accuracy": result["balanced_accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "roc_auc": result["roc_auc"],
            "predictions": result["predictions"],
            "probabilities": result["probabilities"]
        }

    torch.save(existing, results_path)

    print("\n" + "=" * 80)
    print("RESULTS UPDATED")
    print("=" * 80)
    print(results_path)

    return results