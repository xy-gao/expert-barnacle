from collections import Counter

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


class BaseImageGenerator:
    def __init__(self, image_path: str, size: int):
        self.size = (size, size)
        self.img = Image.open(image_path).convert("RGB").resize(self.size)

    def generate_with_dominant_color(self, n_colors=5):
        array = np.array(self.img).reshape(-1, 3)
        clusters = KMeans(n_clusters=n_colors).fit(array)
        colors = clusters.cluster_centers_
        indices = np.random.choice(
            np.array(range(n_colors)), self.size, p=self.cluster_ratio(clusters.labels_)
        )
        img = colors[indices].clip(0, 255)

        return Image.fromarray(np.uint8(img))

    @staticmethod
    def cluster_ratio(labels):
        total = len(labels)
        return [v / total for k, v in sorted(Counter(labels).items())]
