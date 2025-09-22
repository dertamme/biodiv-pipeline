import os
import json
import pandas as pd
import spacy
import nltk
import re
from gensim import corpora
from gensim.models import LdaMulticore, CoherenceModel
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt 



def _check_and_download_nltk_data():
    
    # Prüft, ob die benötigten NLTK-Pakete vorhanden sind, und lädt sie bei Bedarf herunter.
    
    required_packages = ['stopwords', 'wordnet', 'omw-1.4']
    for package in required_packages:
        try:
            nltk.data.find(f'corpora/{package}.zip')
        except LookupError:
            print(f"NLTK-Paket '{package}' nicht gefunden. Lade herunter...")
            nltk.download(package)

def _extrahiere_alle_eintraege(input_ordner: str) -> list[dict]:
    # Liest alle JSON-Dateien und extrahiert Aktionen/Metriken als Liste von Dictionaries.
    alle_eintraege = []
    
    # Schleife über alle Dateien im Input-Ordner.
    for dateiname in os.listdir(input_ordner):
        if not dateiname.lower().endswith(".json"):
            continue

        unternehmen = os.path.splitext(dateiname)[0]
        voller_pfad = os.path.join(input_ordner, dateiname)
        try:
            data = None
            for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
                try:
                    with open(voller_pfad, 'r', encoding=encoding) as f:
                        data = json.load(f)
                    break 
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            if data is None:
                print(f"Warnung: Konnte die Datei {dateiname} nicht lesen. Überspringe.")
                continue

            # Schleife über die Passage-Blöcke.
            for passage_block in data.get("biodiversity_passages", []):
                # Verarbeite Actions
                for action_string in passage_block.get("actions", []):
                    status, text = action_string.split(":", 1) if ":" in action_string else ("Unknown", action_string)
                    alle_eintraege.append({
                        "Unternehmen": unternehmen,
                        "Typ": "Action",
                        "Aussage": text.strip().strip("'\""),
                        "Status": status.strip().capitalize()
                    })
                # Verarbeite Metrics
                for metric_string in passage_block.get("metrics", []): 
                    status, text = metric_string.split(":", 1) if ":" in metric_string else ("Unknown", metric_string)
                    alle_eintraege.append({
                        "Unternehmen": unternehmen,
                        "Typ": "Metric",
                        "Aussage": text.strip().strip("'\""),
                        "Status": status.strip().capitalize()
                    })
        except Exception as e:
            print(f"Fehler beim Lesen von {dateiname}: {e}")
            
    return alle_eintraege

def _preprocess_texts_for_lda(textliste: list[str], nlp, all_stop_words: set) -> list[list[str]]:
    # Bereitet Texte für die LDA-Analyse vor
    prozessierte_texte = []
    print("Verarbeite Texte für LDA (Lemmatisierung, Filterung)...")
    
    # Schleife über alle Texte zur Vorverarbeitung.
    for doc in nlp.pipe(textliste, batch_size=50):
        tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_punct 
            and not token.is_space
            and token.is_alpha
            and len(token.lemma_) > 2
            and token.lemma_.lower() not in all_stop_words
        ]
        prozessierte_texte.append(tokens)
    
    return prozessierte_texte

def _find_optimal_k(corpus, dictionary, texts, speicherordner: str, limit=100):
    # Berechnet den Coherence Score für verschiedene Anzahlen an Themen, um den optimalen Wert zu finden.

    print(f"\nSuche nach der optimalen Anzahl an Themen (bis zu {limit} Themen)...")
    coherence_values = []
    topic_range = range(2, limit + 1, 2)
    
    # Schleife über eine definierte Anzahl an möglichen Themen.
    for num_topics in topic_range:
        print(f"  Teste Modell mit {num_topics} Themen...")
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, workers=4, passes=10, random_state=100)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    # --- Visualisierung und Ausgabe der Ergebnisse ---
    print("\n--- Ergebnisse der Coherence-Analyse ---")
    # Schleife über die Ergebnisse zur Ausgabe in der Konsole.
    for i, score in enumerate(coherence_values):
        print(f"Anzahl Themen: {topic_range[i]}, Coherence Score: {score:.4f}")

    # Erstelle und speichere die Grafik
    plt.figure(figsize=(12, 8))
    plt.plot(topic_range, coherence_values, marker='o')
    plt.title('LDA Coherence Scores vs. Anzahl der Themen', fontsize=16)
    plt.xlabel('Anzahl der Themen (k)')
    plt.ylabel('Coherence Score (höher ist besser)')
    plt.xticks(topic_range)
    plt.grid(True)
    
    plot_path = os.path.join(speicherordner, "lda_coherence_scores.png")
    try:
        plt.savefig(plot_path)
        print(f"\nCoherence-Score-Grafik gespeichert unter: {plot_path}")
    except Exception as e:
        print(f"Fehler beim Speichern der Grafik: {e}")
    plt.close()

    # Finde den Index des höchsten Coherence Scores
    best_result_index = np.argmax(coherence_values)
    optimal_k = topic_range[best_result_index]
    print(f"\nOptimale Anzahl an Themen gefunden: {optimal_k} (Höchster Coherence Score)")
    return optimal_k


def fuehre_lda_analyse_durch(input_ordner: str, speicherordner: str, limit_k_suche: int = 50, min_topic_probability: float = 0.3):
    """
    Führt eine vollständige Themenanalyse mit LDA durch, findet die optimale
    Anzahl an Themen und speichert die Ergebnisse.

    Args:
        min_topic_probability (float): Der Mindest-Wahrscheinlichkeitsscore, damit eine Aussage
                                       einem Thema zugeordnet wird. Andernfalls gilt sie als
                                       "Nicht eindeutig".
    """
    print("--- Beginne Themenanalyse mit Latent Dirichlet Allocation (LDA) ---")
    
    # Erstelle den spezifischen Ausgabeordner
    lda_output_ordner = os.path.join(speicherordner, "LDA")
    os.makedirs(lda_output_ordner, exist_ok=True)

    # --- STUFE 1: DATEN LADEN UND MODELLE EINMALIG INITIALISIEREN ---
    
    _check_and_download_nltk_data()
    print("Lade englisches spaCy-Modell für die Lemmatisierung...")
    try:
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        print("Sprachmodelle geladen.")
    except OSError:
        print("spaCy-Modell 'en_core_web_sm' nicht gefunden.")
        print("Bitte führen Sie aus: python -m spacy download en_core_web_sm")
        return
        
    english_stop_words = set(stopwords.words('english'))
    custom_stop_words = {
        'hectare', 'ha', 'meter', 'm', 'kilometer', 'km', 'ton', 'tonne', 'kg',
        'co2', 'ghg', 'biodiversity', 'company', 'group', 'business', 'report', 'year'
    }
    all_stop_words = english_stop_words.union(custom_stop_words)

    alle_eintraege = _extrahiere_alle_eintraege(input_ordner)
    if not alle_eintraege:
        print("Keine Aktionen oder Metriken gefunden.")
        return
        
    df_main = pd.DataFrame(alle_eintraege)
    # Übergebe die geladenen Modelle an die Vorverarbeitungsfunktion
    preprocessed_docs = _preprocess_texts_for_lda(df_main['Aussage'].tolist(), nlp, all_stop_words)

    # Erstelle das Gensim-Wörterbuch und den Korpus
    dictionary = corpora.Dictionary(preprocessed_docs)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in preprocessed_docs]

    if not corpus:
        print("Korpus ist nach der Filterung leer. Breche Analyse ab.")
        return

    # --- STUFE 2: OPTIMALE THEMENANZAHL FINDEN ---
    optimal_num_topics = _find_optimal_k(corpus, dictionary, preprocessed_docs, lda_output_ordner, limit=limit_k_suche)

    # --- STUFE 3: FINALES LDA-MODELL TRAINIEREN ---
    print(f"\nTrainiere finales LDA-Modell mit {optimal_num_topics} Themen...")
    lda_model = LdaMulticore(corpus=corpus,
                             id2word=dictionary,
                             num_topics=optimal_num_topics,
                             random_state=100,
                             chunksize=100,
                             passes=20, # Mehr Durchläufe > stabileres Ergebnis
                             per_word_topics=True,
                             workers=4)

    # --- STUFE 4: ERGEBNISSE ZUORDNEN UND SPEICHERN ---
    print("\nOrdne Themen den Aussagen zu und erstelle den finalen Report...")
    
    # Erstelle eine Liste der Themennamen basierend auf den Top-Keywords
    topic_names = {}
    # Schleife über alle gefundenen Themen.
    for idx, topic in lda_model.print_topics(-1, num_words=5):
        name = re.sub(r'[^a-zA-Z\s,]', '', topic).replace('  ', ' ').strip()
        topic_names[idx] = name.replace(' ,', ',')

    # Finde für jede Aussage das wahrscheinlichste Thema
    topic_assignments = []
    # Schleife über alle Dokumente im Korpus.
    for doc_bow in corpus:
        if not doc_bow:
            topic_assignments.append(-1) 
            continue
            
        # Hole die Themen und ihre Wahrscheinlichkeiten
        topics = lda_model.get_document_topics(doc_bow, minimum_probability=0.0)
        # Sortiere die Themen nach Wahrscheinlichkeit und nimm das beste
        top_topic = sorted(topics, key=lambda x: x[1], reverse=True)[0]
        
        top_topic_id, top_prob = top_topic
        
        if top_prob >= min_topic_probability:
            topic_assignments.append(top_topic_id)
        else:
            topic_assignments.append(-1)

    df_main['lda_topic_id'] = topic_assignments
    df_main['lda_topic_name'] = df_main['lda_topic_id'].map(topic_names).fillna("Nicht eindeutig")

    # Speichere das finale Ergebnis
    final_path = os.path.join(lda_output_ordner, "lda_themen_report.xlsx")
    df_main[['Unternehmen', 'Typ', 'Aussage', 'Status', 'lda_topic_id', 'lda_topic_name']].to_excel(final_path, index=False)
    
    print(f"\n--- LDA-Analyse vollständig abgeschlossen. ---")
    print(f"Report gespeichert unter: '{final_path}'")
