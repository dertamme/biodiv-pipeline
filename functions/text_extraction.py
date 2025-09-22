import os
import fitz 
import pytesseract
from PIL import Image
import io
import json
import nltk
import re
import spacy
from langdetect import detect, LangDetectException
from functions.status import load_status, save_status


# Stellt sicher, dass alle für die Verarbeitung benötigten NLTK-Pakete vorhanden sind.
def nltlk_setup():
    """Prüft und lädt die notwendigen NLTK-Datenpakete herunter."""
    required_packages = {
        "tokenizers/punkt": "punkt",
        "taggers/averaged_perceptron_tagger": "averaged_perceptron_tagger",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4"
    }
    for path, package_id in required_packages.items():
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"NLTK '{package_id}' Paket nicht gefunden. Lade herunter...")
            nltk.download(package_id, quiet=True)
            print(f"'{package_id}' Paket heruntergeladen.")

nltlk_setup()

from nltk.tokenize import word_tokenize

# Definiert die unterstützten Sprachen und ihre Entsprechungen.
SUPPORTED_LANGUAGES = {
    'en': {'spacy': 'en_core_web_sm', 'nltk': 'english'},
    'de': {'spacy': 'de_core_news_sm', 'nltk': 'german'},
    'fr': {'spacy': 'fr_core_news_sm', 'nltk': 'french'},
    'es': {'spacy': 'es_core_news_sm', 'nltk': 'spanish'},
    'it': {'spacy': 'it_core_news_sm', 'nltk': 'italian'},
    'pt': {'spacy': 'pt_core_news_sm', 'nltk': 'portuguese'},
    'nl': {'spacy': 'nl_core_news_sm', 'nltk': 'dutch'},
    'pl': {'spacy': 'pl_core_news_sm', 'nltk': 'polish'},
    'da': {'spacy': 'da_core_news_sm', 'nltk': 'danish'},
    'fi': {'spacy': 'fi_core_news_sm', 'nltk': 'finnish'},
    'no': {'spacy': 'nb_core_news_sm', 'nltk': 'norwegian'},
    'sv': {'spacy': 'sv_core_news_sm', 'nltk': 'swedish'}
}

# Lädt alle benötigten spaCy-Modelle beim Start des Skripts, um die Performance zu verbessern.
print("Lade spaCy-Sprachmodelle...")
SPACY_MODELS = {
    lang: spacy.load(details['spacy'], disable=["parser", "ner"]) 
    for lang, details in SUPPORTED_LANGUAGES.items()
}
print("Sprachmodelle geladen.")


CURRENT_STAGE_KEY = "text_extraction"


# --- HILFSFUNKTIONEN ---

# Lädt die sprachspezifischen Suchbegriffe 
def lade_suchbegriffe(json_pfad: str) -> dict:
    """Lädt Suchbegriffe aus einer JSON-Datei."""
    try:
        with open(json_pfad, 'r', encoding='utf-8') as f:
            print(f"Lade Suchbegriffe aus '{json_pfad}'...")
            return json.load(f)
    except FileNotFoundError:
        print(f"Fehler: Die Suchbegriff-Datei '{json_pfad}' wurde nicht gefunden.")
        return {}
    except json.JSONDecodeError:
        print(f"Fehler: Die Datei '{json_pfad}' enthält ungültiges JSON.")
        return {}

# Lemmatisiert Text mit dem passenden spaCy-Modell.
def lemmatize_text(text, nlp_model):
    """Lemmatisiert einen Text mit einem geladenen spaCy-Modell."""
    doc = nlp_model(text)
    return " ".join([token.lemma_ for token in doc])

# Bereinigt einen Textstring.
def clean_text(text):
    """Konvertiert Text in Kleinbuchstaben und entfernt Zeilenumbrüche."""
    text = text.lower()
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

# Überprüft die Sprache eines Textes.
def detect_language(text, fallback_lang='unbekannt'):
    """Erkennt die Hauptsprache eines Textes."""
    try:
        return detect(text[:2000])
    except LangDetectException:
        return fallback_lang

# Führt OCR auf einer Seite durch.
def ocr_page_text(page_object, language='deu+eng'):
    """Führt OCR auf einer Seite durch."""
    try:
        pix = page_object.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))
        text = pytesseract.image_to_string(img, lang=language)
        return text
    except Exception:
        return ""



# Verarbeitet PDFs, erkennt die Sprache und führt eine sprachspezifische Analyse durch.
def text_extraction(input_ordner, output_ordner, ocr_verwenden, max_sentence_gap_for_cluster=5):
    
    SUCHBEGRIFFE_JSON_PFAD = "./functions/suchbegriffe.json" 
    alle_suchbegriffe_geladen = lade_suchbegriffe(SUCHBEGRIFFE_JSON_PFAD)
    alle_suchbegriffe=alle_suchbegriffe_geladen

    target_output_dir = os.path.join(output_ordner, "biodiv_text_passages")
    os.makedirs(target_output_dir, exist_ok=True)
    
    # Schleife über alle Dateien im Input-Ordner.
    for dateiname in os.listdir(input_ordner):
        if not dateiname.lower().endswith(".pdf"):
            continue

        if load_status(dateiname, CURRENT_STAGE_KEY):
            continue

        voller_pfad_pdf = os.path.join(input_ordner, dateiname)
        print(f"\n--- Verarbeite Datei: {dateiname} ---")
        
        doc = None
        try:
            doc = fitz.open(voller_pfad_pdf)
            sample_text = "".join([doc.load_page(i).get_text("text") for i in range(min(3, doc.page_count))])
            
            if not sample_text.strip():
                 print(f"  Dokument '{dateiname}' enthält keinen extrahierbaren Text. Überspringe.")
                 save_status(dateiname, CURRENT_STAGE_KEY)
                 continue

            lang_code = detect_language(sample_text)
            
            if lang_code not in SUPPORTED_LANGUAGES:
                print(f"  Dokument '{dateiname}' als '{lang_code}' erkannt. Sprache nicht unterstützt. Überspringe.")
                save_status(dateiname, CURRENT_STAGE_KEY)
                continue
            
            print(f"  Sprache erkannt: {lang_code}")
            nlp = SPACY_MODELS[lang_code]
            nltk_lang = SUPPORTED_LANGUAGES[lang_code]['nltk']

            aktuelle_suchbegriffe = alle_suchbegriffe.get(lang_code)
            if not aktuelle_suchbegriffe:
                print(f"  Keine Suchbegriffe für die Sprache '{lang_code}' in der JSON-Datei gefunden. Überspringe.")
                save_status(dateiname, CURRENT_STAGE_KEY)
                continue

            lemmatized_keywords = [lemmatize_text(clean_text(kw), nlp) for kw in aktuelle_suchbegriffe]
            keyword_regex = re.compile(r"\b(" + "|".join(re.escape(kw) for kw in lemmatized_keywords) + r")\b", re.IGNORECASE)

            alle_saetze_des_dokuments = []
            
            # Schleife über alle Seiten des Dokuments.
            for page in doc:
                page_text_original = ocr_page_text(page) if ocr_verwenden else page.get_text("text")
                if not page_text_original or not page_text_original.strip():
                    continue

                lemmatized_page_text = lemmatize_text(clean_text(page_text_original), nlp)

                if keyword_regex.search(lemmatized_page_text):
                    sentences_on_this_page = nltk.sent_tokenize(page_text_original.replace('\n', ' '), language=nltk_lang)
                    for s in sentences_on_this_page:
                        if s.strip():
                            alle_saetze_des_dokuments.append((s.strip(), page.number + 1))
            
            if not alle_saetze_des_dokuments:
                print(f"Keine relevanten Sätze in '{dateiname}' gefunden.")
                save_status(dateiname, CURRENT_STAGE_KEY) 
                continue

            keyword_sentence_indices = []
            # Schleife über alle gesammelten Sätze zur Index-Findung.
            for i, (original_sentence, _) in enumerate(alle_saetze_des_dokuments):
                lemmatized_sentence = lemmatize_text(clean_text(original_sentence), nlp)
                if keyword_regex.search(lemmatized_sentence):
                    keyword_sentence_indices.append(i)
            
            if not keyword_sentence_indices:
                save_status(dateiname, CURRENT_STAGE_KEY)
                continue

            sentence_clusters = []
            if keyword_sentence_indices:
                current_cluster = [keyword_sentence_indices[0]]
                # Schleife über die Keyword-Indizes zur Cluster-Bildung.
                for i in range(1, len(keyword_sentence_indices)):
                    if keyword_sentence_indices[i] - current_cluster[-1] <= max_sentence_gap_for_cluster:
                        current_cluster.append(keyword_sentence_indices[i])
                    else:
                        sentence_clusters.append(current_cluster)
                        current_cluster = [keyword_sentence_indices[i]]
                sentence_clusters.append(current_cluster)
            
            extrahierte_textbloecke_fuer_diese_pdf = []
            processed_snippets_for_this_pdf = set()

            # Schleife über die Satz-Cluster zur Extraktion.
            for cluster in sentence_clusters:
                first_keyword_idx, last_keyword_idx = cluster[0], cluster[-1]
                start_context_idx = max(0, first_keyword_idx - 5)
                end_context_idx = min(len(alle_saetze_des_dokuments) - 1, last_keyword_idx + 5)
                
                context_window_tuples = alle_saetze_des_dokuments[start_context_idx : end_context_idx + 1]
                focused_passage = " ".join(s_tuple[0] for s_tuple in context_window_tuples).strip()
                
                if focused_passage and focused_passage not in processed_snippets_for_this_pdf:
                    page_numbers = {s_tuple[1] for s_tuple in context_window_tuples}
                    min_page, max_page = min(page_numbers), max(page_numbers)
                    page_range_str = str(min_page) if min_page == max_page else f"{min_page}-{max_page}"
                    
                    # Identifiziere, welche spezifischen Keywords im gefundenen Textabschnitt enthalten sind.
                    found_keywords_in_passage = set()
                    for keyword in aktuelle_suchbegriffe:
                        # Suche case-insensitiv nach dem Keyword im Text
                        if re.search(r'\b' + re.escape(keyword) + r'\b', focused_passage, re.IGNORECASE):
                            found_keywords_in_passage.add(keyword)
                    
                    # Füge Feld "found_keywords" zum Output-Dictionary hinzu.
                    extrahierte_textbloecke_fuer_diese_pdf.append({
                        "page_range": page_range_str, 
                        "passage_text": focused_passage,
                        "found_keywords": list(found_keywords_in_passage) 
                    })
                    processed_snippets_for_this_pdf.add(focused_passage)
            
            if extrahierte_textbloecke_fuer_diese_pdf: 
                basisname_ohne_ext = os.path.splitext(dateiname)[0]
                json_dateipfad = os.path.join(target_output_dir, f"{basisname_ohne_ext}.json")
                with open(json_dateipfad, 'w', encoding='utf-8') as jsonfile:
                    json.dump({"source_pdf": dateiname, "extracted_passages": extrahierte_textbloecke_fuer_diese_pdf}, jsonfile, ensure_ascii=False, indent=4)
                print(f"Textpassagen für '{dateiname}' wurden gespeichert.")
            
            save_status(dateiname, CURRENT_STAGE_KEY)

        except Exception as e:
            print(f"Ein unerwarteter Fehler bei der Verarbeitung der Datei {dateiname} aufgetreten: {e}")
        finally:
            if doc: doc.close()