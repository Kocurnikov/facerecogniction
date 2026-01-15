import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# ===============================
# ŚCIEŻKI
# ===============================
ORIGINAL_TRAIN = 'datasets/original/train'
BALANCED_TRAIN = 'datasets/balanced/train'

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("Generowanie wizualizacji...")
print("=" * 60)

# =============================================================================
# 1. PRZYKŁADOWE OBRAZY DLA KAŻDEJ KLASY
# =============================================================================
print("\n1. Przykładowe obrazy dla każdej klasy...")

for emotion in class_names:
    emotion_path = os.path.join(ORIGINAL_TRAIN, emotion)

    images = [
        f for f in os.listdir(emotion_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ][:5]

    if not images:
        print(f"⚠ Brak obrazów dla klasy {emotion}")
        continue

    fig, axes = plt.subplots(1, len(images), figsize=(14, 3))
    fig.suptitle(f'Emocja: {emotion.upper()}', fontsize=16, fontweight='bold')

    if len(images) == 1:
        axes = [axes]

    for i, img_name in enumerate(images):
        img = load_img(
            os.path.join(emotion_path, img_name),
            color_mode='grayscale',
            target_size=(48, 48)
        )
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')

    fig.subplots_adjust(bottom=0.2)
    fig.savefig(f'viz_1_{emotion}_samples.png', dpi=300)
    plt.close(fig)

    print(f"   ✓ Zapisano: viz_1_{emotion}_samples.png")

# =============================================================================
# 2. ROZKŁAD KLAS – PRZED / PO OVERSAMPLINGU
# =============================================================================
print("\n2. Rozkład klas...")

original_counts = {}
balanced_counts = {}

for emotion in class_names:
    orig_path = os.path.join(ORIGINAL_TRAIN, emotion)
    bal_path = os.path.join(BALANCED_TRAIN, emotion)

    original_counts[emotion] = len([
        f for f in os.listdir(orig_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    balanced_counts[emotion] = (
        len([
            f for f in os.listdir(bal_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        if os.path.exists(bal_path)
        else original_counts[emotion]
    )

    print(f"{emotion}: {original_counts[emotion]} → {balanced_counts[emotion]}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
x = np.arange(len(class_names))

# PRZED
bars1 = ax1.bar(x, [original_counts[c] for c in class_names], color='steelblue')
ax1.set_title("PRZED oversamplingiem", fontweight='bold')
ax1.set_ylabel("Liczba obrazów")
ax1.set_xticks(x)
ax1.set_xticklabels(class_names, rotation=45, ha='right')
ax1.grid(axis='y', alpha=0.3)

for bar in bars1:
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        int(bar.get_height()),
        ha='center',
        va='bottom',
        fontsize=9
    )

# PO
bars2 = ax2.bar(x, [balanced_counts[c] for c in class_names], color='forestgreen')
ax2.set_title("PO oversamplingu", fontweight='bold')
ax2.set_ylabel("Liczba obrazów")
ax2.set_xticks(x)
ax2.set_xticklabels(class_names, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)

for bar in bars2:
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        int(bar.get_height()),
        ha='center',
        va='bottom',
        fontsize=9
    )

fig.subplots_adjust(bottom=0.3, wspace=0.25)
fig.savefig('viz_2_class_distribution.png', dpi=300)
plt.close(fig)

print("   ✓ Zapisano: viz_2_class_distribution.png")

# =============================================================================
# 3. PRZYKŁADY AUGMENTACJI
# =============================================================================
print("\n3. Przykłady augmentacji...")

datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

samples = []

for emotion in class_names:
    emotion_path = os.path.join(ORIGINAL_TRAIN, emotion)
    images = [
        f for f in os.listdir(emotion_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    if images:
        img = load_img(
            os.path.join(emotion_path, images[0]),
            color_mode='grayscale',
            target_size=(48, 48)
        )
        samples.append((emotion, img))

    if len(samples) == 2:
        break

fig, axes = plt.subplots(len(samples), 7, figsize=(18, 5 * len(samples)))
fig.suptitle("Przykłady augmentacji danych", fontsize=18, fontweight='bold')

for row, (emotion, img) in enumerate(samples):
    x_img = img_to_array(img).reshape((1, 48, 48, 1))
    aug_iter = datagen.flow(x_img, batch_size=1)

    axes[row, 0].imshow(img, cmap='gray')
    axes[row, 0].set_title("ORYGINAŁ")
    axes[row, 0].set_xlabel(emotion)
    axes[row, 0].axis('off')

    for col in range(1, 7):
        aug_img = next(aug_iter)[0].squeeze()
        axes[row, col].imshow(aug_img, cmap='gray')
        axes[row, col].set_xlabel(f"Augmentacja {col}")
        axes[row, col].axis('off')

fig.subplots_adjust(top=0.88, hspace=0.4)
fig.savefig('viz_3_augmentation_examples.png', dpi=300)
plt.close(fig)

print("   ✓ Zapisano: viz_3_augmentation_examples.png")
print("\nWszystkie wizualizacje zostały poprawnie wygenerowane ✅")