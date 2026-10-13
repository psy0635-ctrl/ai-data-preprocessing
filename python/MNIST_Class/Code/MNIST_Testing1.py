from pathlib import Path
import numpy as np
from PIL import Image

# 기존 Docker 환경의 이미지 위치
IMAGE_ROOT = Path('/workspace/MNIST/images')


def load_images(root):
    """0~9 폴더의 이미지를 입력 배열과 One-hot 정답으로 변환합니다."""
    images, labels = [], []
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    for number in range(10):
        folder = Path(root) / str(number)

        if not folder.is_dir():
            raise FileNotFoundError(f'숫자 폴더가 없습니다: {folder}')

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in extensions
        )

        if not files:
            raise ValueError(f'이미지 파일이 없습니다: {folder}')

        for path in files:
            with Image.open(path) as image:
                image = image.convert('L')

                if image.size != (28, 28):
                    raise ValueError(
                        f'28×28 이미지가 아닙니다: {path}'
                    )

                array = np.asarray(image, dtype=np.float32)

            # 훈련과 평가에 동일한 -1~1 정규화 적용
            images.append(
                ((array / 127.5) - 1.0).reshape(784)
            )
            labels.append(number)

        print(f'{folder.name} 폴더: {len(files)}개 처리 완료')

    x_data = np.asarray(images, dtype=np.float32)

    # 숫자 정답을 길이 10인 One-hot 배열로 변환
    y_data = np.eye(10, dtype=np.float32)[labels]

    return x_data, y_data


if __name__ == '__main__':
    x_test_datas, y_test_datas = load_images(
        IMAGE_ROOT / 'testing'
    )

    print('평가 입력:', x_test_datas.shape)
    print('평가 정답:', y_test_datas.shape)
    print(
        '정규화 범위:',
        x_test_datas.min(),
        x_test_datas.max()
    )