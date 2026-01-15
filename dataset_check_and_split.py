import kagglehub
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
import random
import shutil

original_path = kagglehub.dataset_download("msambare/fer2013")
print(original_path)

data_dir = Path(original_path)
for f in data_dir.iterdir():
    print(f.name)

train_dir = data_dir / "train"
for f in train_dir.iterdir():
    print(f.name)

test_dir = data_dir / "test"
for f in test_dir.iterdir():
    print(f.name)   



def daily_dataset_check(base_dir):
    """
    Analiza folderu datasetu
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print("Folder nie istnieje:", base_dir)
        return
    
    print(f"\nAnaliza folderu: {base_dir}")
    
    total_files = 0
    all_extensions = []
    missing_suffix = []
    
    # Sprawdź każdy podfolder (klasę)
    for class_folder in base_path.iterdir():
        if class_folder.is_dir():
            files = [f for f in class_folder.iterdir() if f.is_file()]
            total_files += len(files)
            
            # Zlicz rozszerzenia i sprawdź brakujące
            for f in files:
                if f.suffix:
                    all_extensions.append(f.suffix.lower())
                else:
                    missing_suffix.append(f.name)
            
            # Wypisz podstawowe info i kilka przykładów
            print(f"\nKlasa: {class_folder.name}")
            print(f"  Liczba plików: {len(files)}")
            print("  Przykładowe pliki:", [f.name for f in files[:5]])
    
    # Podsumowanie rozszerzeń
    ext_count = Counter(all_extensions)
    print("\nRozszerzenia plików w folderze:")
    for ext, count in ext_count.items():
        print(f"  {ext}: {count} plików")
    
    # Pliki bez rozszerzenia
    if missing_suffix:
        print("\nPliki bez rozszerzenia / błędne:")
        for f in missing_suffix:
            print(" ", f)
    else:
        print("\nWszystkie pliki mają rozszerzenia.")
    
    print(f"\nŁączna liczba plików w folderze: {total_files}")
    print(f"Liczba klas: {len([f for f in base_path.iterdir() if f.is_dir()])}")

# Przykład użycia
train_dir = Path(original_path) / "train"
test_dir = Path(original_path) / "test"
daily_dataset_check(train_dir)
daily_dataset_check(test_dir)


def plot_multiple_dirs_distribution(dir_list, title="File Counts per Class", save_path="distribution.png"):
    """
    Tworzy wykres słupkowy dla dowolnej liczby folderów i zapisuje go do pliku.
    
    dir_list: lista folderów do analizy. Każdy element: (ścieżka, etykieta)
        np. [(train_dir, "Train"), (test_dir, "Test"), (val_dir, "Validation")]
    title: tytuł wykresu
    save_path: ścieżka do pliku PNG, w którym zapisany zostanie wykres
    """
    # Zbierz wszystkie klasy z każdego folderu
    all_classes = set()
    for folder, _label in dir_list:
        folder = Path(folder)
        all_classes.update([f.name for f in folder.iterdir() if f.is_dir()])

    classes = []
    dataset_type = []
    counts = []

    # Zliczamy pliki
    for folder, label in dir_list:
        folder = Path(folder)
        folder_classes = {f.name: len([x for x in f.iterdir() if x.is_file()])
                          for f in folder.iterdir() if f.is_dir()}
        for cls in all_classes:
            classes.append(cls)
            dataset_type.append(label)
            counts.append(folder_classes.get(cls, 0))

    # Tworzymy DataFrame
    df = pd.DataFrame({
        'Class': classes,
        'Dataset': dataset_type,
        'Count': counts
    })

    # Sortujemy klasy według liczby plików w pierwszym folderze
    first_folder_label = dir_list[0][1]
    class_order = df[df['Dataset'] == first_folder_label].sort_values('Count')['Class']

    # Tworzymy wykres
    plt.figure(figsize=(12, 6))
    unique_datasets = df['Dataset'].unique()
    palette = sns.color_palette("Greens", n_colors=len(unique_datasets))
    ax = sns.barplot(
        x='Class',
        y='Count',
        hue='Dataset',
        data=df,
        palette=palette,
        order=class_order
    )
    plt.xticks(rotation=45)
    plt.xlabel('Class')
    plt.ylabel('Number of files')
    plt.title(title, fontsize=16)
    sns.set_style('darkgrid')

    # Dodajemy liczby nad słupkami
    for container in ax.containers:
        ax.bar_label(container, labels=[f'{int(h)}' if h > 0 else '' for h in container.datavalues])

    plt.tight_layout()

    # Zapis do pliku PNG
    plt.savefig(save_path, dpi=300)
    plt.close()  # zamyka figurę, nie wyświetla jej
    print(f"Wykres zapisany do: {save_path}")

dirs_before = [
    (train_dir.resolve(), "Train"),
    (test_dir.resolve(), "Test")]
plot_multiple_dirs_distribution(dirs_before, title="Train_vs_Test_before_split", save_path="Train_vs_Test_before_split.png")

val_dir = data_dir / "val"
val_split = 0.15  # 15% do walidacji

def is_validation_ready(val_dir):
    if not val_dir.exists():
        print("Katalog walidacyjny nie istnieje.")
        return False

    for emotion_folder in val_dir.iterdir():
        if not emotion_folder.is_dir():
            print(f"Znaleziono coś, co nie jest folderem: {emotion_folder}")
            return False
        if not any(f.suffix.lower() in ['.jpg', '.png'] for f in emotion_folder.iterdir() if f.is_file()):
            print(f"Folder {emotion_folder} nie zawiera plików jpg/png.")
            return False

    print("Katalog walidacyjny jest gotowy.")
    return True

# Sprawdzenie katalogu walidacyjnego
if is_validation_ready(val_dir):
    print("Katalog walidacyjny już istnieje i jest poprawnie wypełniony. Nic nie trzeba robić.")
else:
    val_dir.mkdir(parents=True, exist_ok=True)

    for emotion_folder in train_dir.iterdir():
        if not emotion_folder.is_dir():
            continue
        images = [f for f in emotion_folder.iterdir() if f.is_file()]
        random.shuffle(images)
        n_val = int(len(images) * val_split)
        val_images = images[:n_val]

        class_val_dir = val_dir / emotion_folder.name
        class_val_dir.mkdir(parents=True, exist_ok=True)

        for img in val_images:
            shutil.move(str(img), str(class_val_dir / img.name))

dirs_after= [
    (train_dir.resolve(), "Train"),
    (test_dir.resolve(), "Test"),
    (val_dir.resolve(), "Validation")]
plot_multiple_dirs_distribution(dirs_after, title="Train_Test_Validation_after_split", save_path="Train_Test_Validation_distribution.png")
