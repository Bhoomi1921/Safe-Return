import os
import random
import numpy as np
import cv2
from deepface import DeepFace

DATASET_PATH = "data"
AUG_DIR = "data_output"  # your existing augmented folder

# -----------------------------
# Distance computation
# -----------------------------
def compute_distance(img1, img2):
    result = DeepFace.verify(
        img1_path=img1,
        img2_path=img2,
        model_name="ArcFace",
        detector_backend="retinaface",
        distance_metric="cosine",
        enforce_detection=False
    )
    return result["distance"]


# -----------------------------
def generate_distances():

    images = [
        os.path.join(DATASET_PATH, f)
        for f in os.listdir(DATASET_PATH)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    aug_images = [
        os.path.join(AUG_DIR, f)
        for f in os.listdir(AUG_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    genuine = []
    imposter = []

    print("Total original images:", len(images))
    print("Total augmented images:", len(aug_images))

    # =========================
    # ✅ Genuine pairs (USE EXISTING AUGS)
    # =========================
    print("Generating genuine pairs from existing augmentations...")

    pair_count = min(len(images), len(aug_images))

    for idx in range(pair_count):
        try:
            img = images[idx]
            aug_img = aug_images[idx]

            d = compute_distance(img, aug_img)
            genuine.append(d)

        except Exception as e:
            pass

    # =========================
    # ❌ Imposter pairs
    # =========================
    print("Generating imposter pairs...")

    for _ in range(min(300, len(images) * 2)):
        img1, img2 = random.sample(images, 2)
        try:
            d = compute_distance(img1, img2)
            imposter.append(d)
        except:
            pass

    print("Genuine:", len(genuine))
    print("Imposter:", len(imposter))

    np.save("genuine_distances.npy", genuine)
    np.save("imposter_distances.npy", imposter)

if __name__ == "__main__":
    generate_distances()
