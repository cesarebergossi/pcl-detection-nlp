import random
import pandas as pd
import nltk
import torch
from deep_translator import GoogleTranslator
import nlpaug.augmenter.word as naw
from tqdm import tqdm

# Download required NLTK data
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Language choices for back-translation
LANG_CHOICES = ['fr', 'de', 'es']  # French, German, Spanish

def back_translate(text):
    """Performs back-translation through a random language."""
    chosen_lang = random.choice(LANG_CHOICES)

    try:
        translated = GoogleTranslator(source='en', target=chosen_lang).translate(text)
        back_translated = GoogleTranslator(source=chosen_lang, target='en').translate(translated)
        print(f"1. Back-translated ({chosen_lang}): {back_translated}")
        return back_translated
    except Exception as e:
        print(f"Back-translation failed: {e}")
        return None

def synonym_replace(text):
    """Uses WordNet synonyms to replace words."""
    aug = naw.SynonymAug(aug_src='wordnet', aug_p=0.2)
    try:
        augmented = aug.augment(text)
        if isinstance(augmented, list):
            augmented_text = augmented[0]
            print(f"2. Synonym Replacement: {augmented_text}")
            return augmented_text
    except Exception as e:
        print(f"Synonym augmentation failed: {e}")
        return None

def masked_lm_augmentation(text):
    """Uses a masked language model (like BERT) to replace words with contextually appropriate alternatives."""
    
    # Move the model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action="substitute", aug_p=0.2, device=device)

    try:
        augmented = aug.augment(text)
        if isinstance(augmented, list):
            augmented_text = augmented[0]
            print(f"3. BERT Augmentation: {augmented_text}")
            return augmented_text
    except Exception as e:
        print(f"BERT augmentation failed: {e}")
        return None

def augment_pcl_data(input_csv, output_csv, sample_fraction=0.4):
    """Augment a subset of the dataset while maintaining class distribution."""
    df = pd.read_csv(input_csv)

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    # Separate PCL and non-PCL samples
    pcl_samples = df[df["label"] == 1]  
    non_pcl_samples = df[df["label"] == 0]

    # Sample 40% of each group
    pcl_sampled = pcl_samples.sample(frac=sample_fraction, random_state=42)
    non_pcl_sampled = non_pcl_samples.sample(frac=sample_fraction, random_state=42)

    # Combine to get the final dataset to augment
    sampled_df = pd.concat([pcl_sampled, non_pcl_sampled])

    augmented_texts = []
    
    # Progress bar
    for _, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing Samples", unit="sample"):
        original_text = row["text"]

        # Apply augmentation methods
        back_translated_text = back_translate(original_text)
        synonym_text = synonym_replace(original_text)

        # Collect valid augmentations
        for new_text in [back_translated_text, synonym_text]:
            if new_text:
                augmented_texts.append({
                    "par_id": row["par_id"], "community": row["community"],
                    "text": new_text, "label": row["label"]
                })

    # Convert to DataFrame
    augmented_df = pd.DataFrame(augmented_texts)

    # Append to original dataset and save
    full_df = pd.concat([df, augmented_df], ignore_index=True)
    full_df.to_csv(output_csv, index=False)

    print(f"Augmentation complete. New dataset saved to {output_csv}")

# Run augmentation when executed as a script
if __name__ == "__main__":
    input_csv = "Data/train_split.csv"
    output_csv = "Data/augmented_train_split.csv"
    augment_pcl_data(input_csv, output_csv)