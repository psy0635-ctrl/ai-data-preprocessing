"""Assignment 3: controlled Iris augmentation / dropout experiments.

Run with .venv/Scripts/python.exe Iris_assignment3.py
Original Flower.py and Flower2.py are preserved as image-classification references.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import argparse
import csv
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
CONDITIONS = {"baseline": (False, False), "augmentation": (True, False),
              "dropout": (False, True), "both": (True, True)}
LABELS = {"baseline": "Baseline", "augmentation": "Augmentation",
          "dropout": "Dropout", "both": "Augmentation + Dropout"}
SEEDS = [123, 456, 789, 2026, 42]


def build_model(seed, augmentation=False, dropout=False):
    """Use identical seeded Dense weights in all four conditions."""
    layers = [keras.Input(shape=(4,))]
    if augmentation:
        layers.append(keras.layers.GaussianNoise(0.1, seed=seed + 20))
    layers.append(keras.layers.Dense(32, activation="relu", name="hidden1",
                  kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 1)))
    if dropout:
        layers.append(keras.layers.Dropout(0.2, seed=seed + 30))
    layers.append(keras.layers.Dense(16, activation="relu", name="hidden2",
                  kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 2)))
    if dropout:
        layers.append(keras.layers.Dropout(0.2, seed=seed + 31))
    layers.append(keras.layers.Dense(3, activation="softmax", name="output",
                  kernel_initializer=keras.initializers.GlorotUniform(seed=seed + 3)))
    model = keras.Sequential(layers)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def dataset(x, y, seed=None):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if seed is not None:
        ds = ds.shuffle(len(y), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    options.experimental_deterministic = True
    return ds.batch(16).with_options(options).prefetch(1)


def plot_results(runs, output):
    metrics = [("accuracy", "Training Accuracy"), ("val_accuracy", "Validation Accuracy"),
               ("loss", "Training Loss"), ("val_loss", "Validation Loss")]
    colors = ["#24476b", "#b27222", "#38816d", "#9c5063"]
    for name in CONDITIONS:
        h = runs[0]["histories"][name]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
            epochs = np.arange(1, len(h[metric]) + 1)
            ax.plot(epochs, h[metric], label="Training", color=colors[0])
            ax.plot(epochs, h["val_" + metric], label="Validation", color=colors[1])
            ax.set(xlabel="Epoch", ylabel=title, title=title)
            if metric == "accuracy": ax.set_ylim(0, 1.03)
            ax.grid(alpha=0.2)
            ax.legend()
        fig.suptitle(f"{LABELS[name]} | Iris | seed={runs[0]['seed']}")
        fig.tight_layout()
        fig.savefig(output / f"history_{name}.png", dpi=180)
        plt.close(fig)
    for averaged in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for ax, (metric, title) in zip(axes.flat, metrics):
            for color, name in zip(colors, CONDITIONS):
                values = np.array([run["histories"][name][metric] for run in runs])
                mean = values.mean(axis=0) if averaged else values[0]
                epochs = np.arange(1, len(mean) + 1)
                ax.plot(epochs, mean, label=LABELS[name], color=color)
                if averaged:
                    sd = values.std(axis=0, ddof=1) if len(runs) > 1 else np.zeros_like(mean)
                    ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.12)
                ax.set(xlabel="Epoch", ylabel="Accuracy" if "accuracy" in metric else "Loss", title=title)
            if "accuracy" in metric: ax.set_ylim(0, 1.03)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.suptitle("Iris: mean +/- sample SD across seeds" if averaged else f"Iris: four conditions | seed={runs[0]['seed']}")
        fig.tight_layout()
        fig.savefig(output / ("comparison_mean.png" if averaged else "comparison_primary.png"), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--output", type=Path, default=ROOT / "iris_results")
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("epochs must be positive and seeds must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    iris = load_iris()
    runs, rows = [], []
    for seed in args.seeds:
        train_idx, val_idx = train_test_split(np.arange(150), test_size=0.2,
                                             stratify=iris.target, random_state=seed)
        scaler = StandardScaler().fit(iris.data[train_idx])
        x_train = scaler.transform(iris.data[train_idx]).astype("float32")
        x_val = scaler.transform(iris.data[val_idx]).astype("float32")
        y_train, y_val = iris.target[train_idx], iris.target[val_idx]
        assert set(train_idx).isdisjoint(val_idx)
        assert np.bincount(y_train).tolist() == [40, 40, 40]
        assert np.bincount(y_val).tolist() == [10, 10, 10]
        assert np.allclose(scaler.mean_, iris.data[train_idx].mean(axis=0))
        run = {"seed": seed, "train_indices": train_idx.tolist(), "validation_indices": val_idx.tolist(),
               "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
               "histories": {}, "evaluations": {}}
        reference = None
        for name, (aug, drop) in CONDITIONS.items():
            keras.backend.clear_session()
            keras.utils.set_random_seed(seed)
            model = build_model(seed, aug, drop)
            weights = [w.numpy().copy() for layer in model.layers
                       if isinstance(layer, keras.layers.Dense) for w in layer.weights]
            if reference is None: reference = weights
            else:
                assert all(np.array_equal(a, b) for a, b in zip(reference, weights))
            # Augmentation and dropout must have no effect at inference time.
            if aug or drop:
                base = build_model(seed)
                np.testing.assert_allclose(model(x_val, training=False).numpy(),
                                           base(x_val, training=False).numpy(), atol=1e-7)
                del base
            print(f"START seed={seed} condition={name}", flush=True)
            h = model.fit(dataset(x_train, y_train, seed), validation_data=dataset(x_val, y_val),
                          epochs=args.epochs, verbose=0, shuffle=False).history
            assert set(h) >= {"accuracy", "val_accuracy", "loss", "val_loss"}
            for key, values in h.items():
                assert len(values) == args.epochs and np.isfinite(values).all(), (name, key)
            train_eval = model.evaluate(dataset(x_train, y_train), verbose=0, return_dict=True)
            val_eval = model.evaluate(dataset(x_val, y_val), verbose=0, return_dict=True)
            np.testing.assert_allclose(val_eval["accuracy"], h["val_accuracy"][-1], atol=1e-6)
            np.testing.assert_allclose(val_eval["loss"], h["val_loss"][-1], atol=1e-6)
            run["histories"][name] = {k: [float(v) for v in vs] for k, vs in h.items()}
            run["evaluations"][name] = {"clean_train": train_eval, "validation": val_eval}
            rows.append({"seed": seed, "condition": name,
                         **{k: float(v[-1]) for k, v in h.items()},
                         "clean_train_accuracy": train_eval["accuracy"],
                         "clean_train_loss": train_eval["loss"]})
            if seed == args.seeds[0]:
                model.save(args.output / f"{name}.keras")
            print(f"DONE {name}: val_acc={val_eval['accuracy']:.4f}, val_loss={val_eval['loss']:.4f}", flush=True)
        runs.append(run)
    summary = {}
    for name in CONDITIONS:
        selected = [r for r in rows if r["condition"] == name]
        summary[name] = {}
        for key in ("accuracy", "val_accuracy", "loss", "val_loss", "clean_train_accuracy", "clean_train_loss"):
            values = [r[key] for r in selected]
            summary[name][key] = {"mean": float(np.mean(values)),
                                  "sample_sd": float(np.std(values, ddof=1)) if len(values)>1 else 0.0}
    result = {"dataset": "sklearn.datasets.load_iris", "features": iris.feature_names,
              "classes": iris.target_names.tolist(), "epochs": args.epochs, "batch_size": 16,
              "seeds": args.seeds, "architecture": [4, 32, 16, 3], "learning_rate": 0.001,
              "noise_stddev": 0.1, "dropout_rate": 0.2,
              "baseline_origin": "New Iris MLP; no pre-existing Iris model found in project",
              "versions": {"python": platform.python_version(), "tensorflow": tf.__version__,
                           "keras": keras.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
              "runs": runs, "summary": summary}
    (args.output / "training_history.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output / "final_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_results(runs, args.output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
