import argparse
import json
import platform
import time
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_digits
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train(output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, y = load_digits(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, stratify=y, random_state=seed)
    baseline = DummyClassifier(strategy='most_frequent').fit(x_train, y_train)
    model = make_pipeline(StandardScaler(), SVC(C=10, gamma='scale'))
    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_seconds = time.perf_counter() - start
    predicted = model.predict(x_test)
    metrics = dict(dataset='sklearn bundled UCI optical digits (8x8)', seed=seed,
        train_samples=len(y_train), test_samples=len(y_test),
        baseline_accuracy=accuracy_score(y_test, baseline.predict(x_test)),
        accuracy=accuracy_score(y_test, predicted), training_seconds=training_seconds,
        sklearn_version=sklearn.__version__, python_version=platform.python_version(),
        platform=platform.platform(), model='StandardScaler + SVC(C=10, gamma=scale)',
        report=classification_report(y_test, predicted, output_dict=True))
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    joblib.dump(model, output/'model.joblib')
    ConfusionMatrixDisplay.from_predictions(y_test, predicted, cmap='Blues')
    plt.title('Held-out digits classification · seed '+str(seed))
    plt.tight_layout()
    plt.savefig(output/'confusion-matrix.png', dpi=140)
    plt.close()
    fig, axes = plt.subplots(2, 6, figsize=(10, 4))
    for index, ax in enumerate(axes.flat):
        ax.imshow(x_test[index].reshape(8, 8), cmap='gray_r')
        ax.set_title(f'True {y_test[index]} / Pred {predicted[index]}')
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(output/'predictions.png', dpi=140)
    plt.close(fig)
    print(json.dumps({k:v for k,v in metrics.items() if k != 'report'}, indent=2))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CPU classification baseline, not LLM fine-tuning')
    parser.add_argument('--output', default='results')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    train(args.output, args.seed)
