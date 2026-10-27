from pathlib import Path
import json

import matplotlib

# Linux 컨테이너에서도 그래프를 파일로 저장할 수 있도록 설정
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. 기본 설정
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "results"
DATA_CACHE_DIR = BASE_DIR / "data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 123
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
EPOCHS = 10

EXPECTED_CLASSES = [
    "daisy",
    "dandelion",
    "roses",
    "sunflowers",
    "tulips",
]

keras.utils.set_random_seed(SEED)


# 2. 데이터 다운로드 및 실제 이미지 폴더 확인
def download_dataset():
    dataset_url = (
        "https://storage.googleapis.com/"
        "download.tensorflow.org/example_images/flower_photos.tgz"
    )

    downloaded_path = keras.utils.get_file(
        fname="flower_photos.tgz",
        origin=dataset_url,
        extract=True,
        cache_dir=str(DATA_CACHE_DIR),
    )

    # Keras 버전에 따라 반환되는 경로가 다를 수 있어
    # 전용 데이터 폴더 안에서 실제 이미지 폴더를 확인한다.
    search_root = DATA_CACHE_DIR / "datasets"

    candidates = [
        Path(downloaded_path),
        search_root / "flower_photos",
    ]

    candidates.extend(
        path
        for path in search_root.rglob("flower_photos")
        if path.is_dir()
    )

    for candidate in candidates:
        if all(
            (candidate / name).is_dir()
            for name in EXPECTED_CLASSES
        ):
            return candidate

    raise FileNotFoundError(
        "꽃 이미지 폴더를 찾지 못했습니다. "
        f"다운로드 및 압축 해제 상태를 확인하세요: {search_root}"
    )


# 3. 학습·검증 데이터 생성
def load_datasets(data_dir):
    common_options = {
        "directory": str(data_dir),
        "validation_split": 0.2,
        "seed": SEED,
        "image_size": (IMG_HEIGHT, IMG_WIDTH),
        "batch_size": BATCH_SIZE,
        "label_mode": "int",
        "color_mode": "rgb",
        "shuffle": True,
    }

    train_ds = keras.utils.image_dataset_from_directory(
        subset="training",
        **common_options,
    )

    val_ds = keras.utils.image_dataset_from_directory(
        subset="validation",
        **common_options,
    )

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names

# 4. 원본 이미지 9장 저장
def save_sample_images(train_ds, class_names):
    images, labels = next(iter(train_ds))

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for index, ax in enumerate(axes.flat):
        if index < int(images.shape[0]):
            ax.imshow(images[index].numpy().astype("uint8"))
            label_index = int(labels[index].numpy())
            ax.set_title(class_names[label_index])

        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "01_sample_images.png",
        dpi=150,
    )
    plt.close(fig)

    print("이미지 배치 크기:", images.shape)
    print("레이블 배치 크기:", labels.shape)

    normalized_images = layers.Rescaling(1.0 / 255)(images)

    print(
        "정규화 후 첫 이미지의 최솟값·최댓값:",
        float(tf.reduce_min(normalized_images[0]).numpy()),
        float(tf.reduce_max(normalized_images[0]).numpy()),
    )


# 5. 데이터 증강 계층
def create_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )


def save_augmented_images(train_ds):
    images, _ = next(iter(train_ds))
    sample = images[:1]

    augmentation = create_augmentation()
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for ax in axes.flat:
        # 학습 모드로 호출하여 실제 증강을 적용한다.
        augmented = augmentation(sample, training=True)

        ax.imshow(
            np.clip(augmented[0].numpy(), 0, 255).astype("uint8")
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "03_augmented_images.png",
        dpi=150,
    )
    plt.close(fig)


# 6. 기본 모델 및 개선 모델 구성
def build_model(num_classes, improved=False):
    model_layers = [
        keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    ]

    if improved:
        model_layers.append(create_augmentation())

    model_layers.extend(
        [
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(
                16, 3, padding="same", activation="relu"
            ),
            layers.MaxPooling2D(),
            layers.Conv2D(
                32, 3, padding="same", activation="relu"
            ),
            layers.MaxPooling2D(),
            layers.Conv2D(
                64, 3, padding="same", activation="relu"
            ),
            layers.MaxPooling2D(),
        ]
    )

    if improved:
        model_layers.append(layers.Dropout(0.2))

    model_layers.extend(
        [
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes),
        ]
    )

    model = keras.Sequential(
        model_layers,
        name="improved_cnn" if improved else "baseline_cnn",
    )

    model.compile(
        optimizer="adam",
        loss=keras.losses.SparseCategoricalCrossentropy(
            from_logits=True
        ),
        metrics=["accuracy"],
    )

    return model


# 7. 학습 결과 그래프 저장
def save_history_plot(history, title, filename):
    values = history.history
    epochs = range(1, len(values["accuracy"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(
        epochs, values["accuracy"], label="Training"
    )
    axes[0].plot(
        epochs, values["val_accuracy"], label="Validation"
    )
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epochs, values["loss"], label="Training"
    )
    axes[1].plot(
        epochs, values["val_loss"], label="Validation"
    )
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)


def save_comparison_plot(baseline_history, improved_history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    histories = [
        ("Baseline", baseline_history),
        ("Augmentation + Dropout", improved_history),
    ]

    for label, history in histories:
        values = history.history
        epochs = range(1, len(values["val_accuracy"]) + 1)

        axes[0].plot(
            epochs, values["val_accuracy"], label=label
        )
        axes[1].plot(
            epochs, values["val_loss"], label=label
        )

    axes[0].set_title("Validation Accuracy")
    axes[0].set_ylabel("Accuracy")
    axes[1].set_title("Validation Loss")
    axes[1].set_ylabel("Loss")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "05_model_comparison.png",
        dpi=150,
    )
    plt.close(fig)


# 8. 전체 실행
def main():
    print("TensorFlow 버전:", tf.__version__)
    print("사용 가능한 GPU:", tf.config.list_physical_devices("GPU"))

    data_dir = download_dataset()

    image_count = len(list(data_dir.glob("*/*.jpg")))

    print("\n데이터 경로:", data_dir)
    print("전체 JPG 이미지 수:", image_count)

    train_ds, val_ds, class_names = load_datasets(data_dir)

    print("꽃 종류:", class_names)

    save_sample_images(train_ds, class_names)
    save_augmented_images(train_ds)

    # 전체 데이터를 메모리에 캐시하지 않고,
    # 다음 배치 하나만 미리 준비한다.
    train_ds = train_ds.prefetch(1)
    val_ds = val_ds.prefetch(1)

    print("\n===== 기본 모델 학습 =====")

    keras.utils.set_random_seed(SEED)
    baseline_model = build_model(
        num_classes=len(class_names),
        improved=False,
    )
    baseline_model.summary()

    baseline_history = baseline_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    save_history_plot(
        baseline_history,
        "Baseline CNN",
        "02_baseline_history.png",
    )

    print("\n===== 개선 모델 학습 =====")

    keras.utils.set_random_seed(SEED)
    improved_model = build_model(
        num_classes=len(class_names),
        improved=True,
    )
    improved_model.summary()

    improved_history = improved_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
    )

    save_history_plot(
        improved_history,
        "CNN with Augmentation and Dropout",
        "04_improved_history.png",
    )

    save_comparison_plot(
        baseline_history,
        improved_history,
    )

    histories = {
        "class_names": class_names,
        "baseline": {
            key: [float(value) for value in values]
            for key, values in baseline_history.history.items()
        },
        "improved": {
            key: [float(value) for value in values]
            for key, values in improved_history.history.items()
        },
    }

    with (OUTPUT_DIR / "training_history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            histories,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== 마지막 Epoch 결과 =====")

    for name, history in [
        ("기본 모델", baseline_history),
        ("개선 모델", improved_history),
    ]:
        values = history.history

        print(
            f"{name}: "
            f"학습 정확도={values['accuracy'][-1]:.4f}, "
            f"검증 정확도={values['val_accuracy'][-1]:.4f}, "
            f"검증 손실={values['val_loss'][-1]:.4f}"
        )

    print("\n실행 완료")
    print("결과 저장 위치:", OUTPUT_DIR)


if __name__ == "__main__":
    main()