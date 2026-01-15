import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, save_img

# Konfiguracja
INPUT_DIR = 'datasets/original/train'
OUTPUT_DIR = 'datasets/augmented/train'
NUM_AUGMENTATIONS = 1

# Augmentacje
datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

# Przetwarzaj wszystkie klasy
for emotion in os.listdir(INPUT_DIR):
    input_path = os.path.join(INPUT_DIR, emotion)
    output_path = os.path.join(OUTPUT_DIR, emotion)
    
    if not os.path.isdir(input_path):
        continue
        
    os.makedirs(output_path, exist_ok=True)
    
    # Przetwarzaj wszystkie obrazy
    for img_name in os.listdir(input_path):
        if not img_name.endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        # Wczytaj obraz
        img = load_img(os.path.join(input_path, img_name), color_mode='grayscale', target_size=(48, 48))
        x = img_to_array(img).reshape((1, 48, 48, 1))
        base_name = os.path.splitext(img_name)[0]
        
        # Zapisz oryginał
        img.save(os.path.join(output_path, f"{base_name}_orig.png"))
        
        # Generuj augmentacje
        i = 0
        for batch in datagen.flow(x, batch_size=1):
            save_img(os.path.join(output_path, f"{base_name}_aug{i}.png"), batch[0])
            i += 1
            if i >= NUM_AUGMENTATIONS:
                break

print(f"✅ Gotowe! Dane zapisane w {OUTPUT_DIR}")