"""
=============================================================================
SKRYPT OVERSAMPLING - Balansowanie niezbalansowanych klas FER2013
=============================================================================

PROBLEM:
Dataset FER2013 jest niezbalansowany:
- disgust: 316 obrazów (NAJMNIEJ)
- happy: 5214 obrazów (NAJWIĘCEJ)
Model ma trudności z nauczeniem się rzadkich klas.

ROZWIĄZANIE:
Inteligentny oversampling - dogenerowanie augmentowanych wersji dla 
klas z małą liczbą przykładów.

LOGIKA:
1. TARGET = 3000 obrazów (cel dla każdej klasy)
2. Klasy < 3000: generuj więcej augmentacji
   - disgust (316) → 8 augmentacji → ~2844 obrazów
3. Klasy ≥ 3000: minimalna augmentacja (1x)
   - happy (5214) → 1 augmentacja → ~10428 obrazów

AUGMENTACJE UŻYTE:
- Przesunięcie poziome/pionowe ±10%
- Odbicie lustrzane (horizontal flip)
- BEZ rotacji/zoom/brightness (mogą zmienić percepcję emocji)

WYNIK:
- Dataset: 20,750 → ~43,600 obrazów
- Lepsze wyniki dla rzadkich klas (disgust, fear)
- Test accuracy: 56.1% (vs 54.8% bez oversampligu)

AUTOR: Tomasz Bartkowski
DATA: 2026-01-05
=============================================================================
"""
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img
import numpy as np

# Konfiguracja
INPUT_DIR = 'datasets/original/train'
OUTPUT_DIR = 'datasets/balanced/train'

# Augmentacje
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Aktualne rozkłady klas (z Twojego wykresu)
class_counts = {
    'angry': 2887,
    'disgust': 316,      # NAJMNIEJ - potrzeba mocnego oversampligu
    'fear': 2961,
    'happy': 5214,       # NAJWIĘCEJ
    'neutral': 3588,
    'sad': 3491,
    'surprise': 2292
}

# Target: ~3000 dla wszystkich (poniżej happy, żeby nie za dużo danych)
TARGET_COUNT = 3000

# Oblicz ile augmentacji potrzeba dla każdej klasy
augmentations_needed = {}
for emotion, count in class_counts.items():
    if count < TARGET_COUNT:
        # Ile augmentacji potrzeba (bez oryginału)
        augmentations_needed[emotion] = max(1, int((TARGET_COUNT - count) / count))
    else:
        # Klasy z dużą liczbą - tylko 1 aug (jak dotychczas)
        augmentations_needed[emotion] = 1

print("Plan oversampligu:")
for emotion, aug_count in augmentations_needed.items():
    original = class_counts[emotion]
    target = original * (aug_count + 1) if aug_count > 0 else original
    print(f"  {emotion:10s}: {original:4d} → ~{target:4d} obrazów (aug={aug_count})")

print(f"\n{'='*60}")
print("Rozpoczynam oversampling...")
print(f"{'='*60}\n")

# Przetwarzaj każdą klasę
for emotion in os.listdir(INPUT_DIR):
    input_path = os.path.join(INPUT_DIR, emotion)
    output_path = os.path.join(OUTPUT_DIR, emotion)
    
    if not os.path.isdir(input_path):
        continue
    
    os.makedirs(output_path, exist_ok=True)
    
    # Pobierz wszystkie obrazy
    images = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    num_augmentations = augmentations_needed.get(emotion, 1)
    
    print(f"Przetwarzanie '{emotion}': {len(images)} obrazów, {num_augmentations} augmentacji...")
    
    total_saved = 0
    
    # Przetwarzaj każdy obraz
    for img_name in images:
        img_path = os.path.join(input_path, img_name)
        
        # Wczytaj obraz
        img = load_img(img_path, color_mode='grayscale', target_size=(48, 48))
        x = img_to_array(img).reshape((1, 48, 48, 1))
        base_name = os.path.splitext(img_name)[0]
        
        # Zapisz oryginał
        img.save(os.path.join(output_path, f"{base_name}_orig.png"))
        total_saved += 1
        
        # Generuj augmentacje
        aug_count = 0
        for batch in datagen.flow(x, batch_size=1):
            save_img(os.path.join(output_path, f"{base_name}_aug{aug_count}.png"), batch[0])
            total_saved += 1
            aug_count += 1
            if aug_count >= num_augmentations:
                break
    
    print(f"  ✅ {emotion}: {total_saved} obrazów zapisanych\n")

print(f"{'='*60}")
print("PODSUMOWANIE OVERSAMPLIGU")
print(f"{'='*60}")

# Policz finalne rozkłady
final_counts = {}
for emotion in os.listdir(OUTPUT_DIR):
    emotion_path = os.path.join(OUTPUT_DIR, emotion)
    if os.path.isdir(emotion_path):
        count = len([f for f in os.listdir(emotion_path) if f.endswith('.png')])
        final_counts[emotion] = count

print("\nFinalne rozkłady klas:")
for emotion in sorted(final_counts.keys()):
    original = class_counts.get(emotion, 0)
    final = final_counts[emotion]
    change = ((final - original) / original * 100) if original > 0 else 0
    print(f"  {emotion:10s}: {original:4d} → {final:4d} (+{change:5.1f}%)")

total_original = sum(class_counts.values())
total_final = sum(final_counts.values())
print(f"\n  RAZEM:       {total_original:4d} → {total_final:4d} obrazów")
print(f"\n✅ Dataset zapisany w: {OUTPUT_DIR}")
print(f"{'='*60}")