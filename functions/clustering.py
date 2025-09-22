import os
import json
import itertools
import numpy as np
import pandas as pd
import hdbscan
from sentence_transformers import SentenceTransformer
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re 
from dotenv import load_dotenv
import google.generativeai as genai
import time
from functions.prompts import gemini_model_version

load_dotenv()

# Stellt sicher, dass das NLTK-Datenpaket 'stopwords' vorhanden ist.
try:
    from nltk.corpus import stopwords
except LookupError:
    print("NLTK-Datenpaket 'stopwords' nicht gefunden. Lade es herunter...")
    nltk.download('stopwords')
    from nltk.corpus import stopwords

CLASSIFY_NOISE_PROMPT = """
Analyze the following corporate statement. Determine if it describes a concrete, specific biodiversity action or if it is a general, non-specific statement without a clear, direct impact on biodiversity.

Respond with one of two possible keywords *only*:
- "Konkrete Massnahme" if it describes a specific action (e.g., "planting 100 trees", "restoring 5 hectares of wetland", "installing green roofs").
- "Keine konkrete Massnahme" if it is a general statement (e.g., "we value biodiversity", "we are committed to nature", "biodiversity is important").

Statement: "{statement}"

Classification:
"""

GROUP_CATEGORIES_PROMPT = """
You are a senior data analyst. Your task is to group a list of granular, keyword-based topics into a meaningful number of high-level super-categories.
The super-category names should be distinct, descriptive, and reflect a strategic business perspective (e.g., "Supply Chain & Sourcing", "Corporate Strategy & Reporting", "Operational Site Management").
Respond with a single JSON object. The keys of the object must be the original granular topic names from the list, and the value for each key must be the assigned super-category name. Ensure every single granular topic from the list is a key in your JSON response.

Here is the list of granular topics to group:
{category_list}

JSON Response:
"""

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
                for metric_string in passage_block.get("summarized_metrics", []):
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

def _lemmatisiere_texte(textliste: list[str]) -> list[str]:
    # Lemmatisiert. 
    print("Lade englisches spaCy-Modell für die Lemmatisierung...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("spaCy-Modell 'en_core_web_sm' nicht gefunden.")
        print("Bitte führen Sie aus: python -m spacy download en_core_web_sm")
        return textliste

    # Benutzerdefinierte Liste von Einheiten und anderen Wörtern, die entfernt werden sollen.
    custom_stop_words = {
        'hectare', 'hectares', 'ha', 'meter', 'meters', 'm', 'kilometer', 'kilometers', 'km',
        'ton', 'tons', 'tonne', 'tonnes', 'kg', 'kilogram', 'kilograms',
        'co2', 'ghg', 'biodiversity'
    }

    lemmatisierte_texte = []
    print("Lemmatisiere Texte und entferne Zahlen/Einheiten/Orte/Daten...")
    
    # Schleife über alle Aktionen zur Lemmatisierung und Bereinigung.
    for doc in nlp.pipe(textliste):
        lemmatisierte_tokens = [
            token.lemma_.lower() for token in doc 
            if not token.is_punct 
            and not re.search(r'\d', token.text) # Entfernt jeden Token, der eine Ziffer enthält
            and token.lower_ not in custom_stop_words
            and token.ent_type_ not in ['DATE', 'GPE', 'LOC', 'FAC', 'ORG', 'MONEY', 'QUANTITY', 'CARDINAL', 'PERCENT', 'ORDINAL']
        ]
        lemmatisierte_texte.append(" ".join(lemmatisierte_tokens))
    
    return lemmatisierte_texte

def _name_clusters_with_keywords(df: pd.DataFrame, group_id_col: str, text_col: str, anzahl_keywords: int) -> dict:
    """
    Generiert Keyword-basierte Namen für eine beliebige Gruppierungsspalte.
    """
    df_filtered = df[df[group_id_col] != -1]
    docs_pro_gruppe = df_filtered.groupby(group_id_col)[text_col].apply(lambda x: ' '.join(x))

    if docs_pro_gruppe.empty: return {}

    min_df_value = 1
    max_df_value = 0.9
    english_stop_words = stopwords.words('english')
    vectorizer = TfidfVectorizer(stop_words=english_stop_words, max_df=max_df_value, min_df=min_df_value)
    
    try:
        tfidf_matrix = vectorizer.fit_transform(docs_pro_gruppe)
    except ValueError:
        print(f"Warnung: Konnte keine Keywords für die Spalte '{group_id_col}' extrahieren.")
        return {gruppen_id: "Keine Keywords gefunden" for gruppen_id in docs_pro_gruppe.index}

    feature_names = vectorizer.get_feature_names_out()
    namen_mapping = {}
    # Schleife über jede Gruppe, um die Top-Keywords zu extrahieren.
    for i, gruppen_id in enumerate(docs_pro_gruppe.index):
        feature_index = tfidf_matrix[i,:].nonzero()[1]
        tfidf_scores = zip(feature_index, [tfidf_matrix[i, x] for x in feature_index])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        top_keywords = [feature_names[i] for i, s in sorted_scores[:anzahl_keywords]]
        cluster_name = ', '.join(top_keywords).capitalize()
        namen_mapping[gruppen_id] = cluster_name if cluster_name else "Nicht spezifiziert"
        
    return namen_mapping

def _get_api_response(prompt: str) -> str:
    """Universelle Funktion für einen einzelnen API-Aufruf."""
    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key: return "FEHLER: GOOGLE_API_KEY nicht gefunden."
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(gemini_model_version)
        response = model.generate_content(prompt)
        
        return response.text.strip().replace("```json", "").replace("```", "").strip()
    except Exception as e:
        print(f"    Fehler bei API-Aufruf: {e}")
        return "API Fehler"

def fuehre_vollstaendige_analyse_durch(input_ordner: str, speicherordner: str):

    print("--- Beginne vollständige, optimierte, hierarchische Cluster-Analyse ---")
    os.makedirs(speicherordner, exist_ok=True)

    # --- STUFE 1: DATEN LADEN UND VORBEREITEN ---
    alle_eintraege = _extrahiere_alle_eintraege(input_ordner)
    if not alle_eintraege:
        print("Keine Aktionen oder Metriken in den JSON-Dateien gefunden.")
        return
        
    df_main = pd.DataFrame(alle_eintraege)
    df_main['aktion_lemmatisiert'] = _lemmatisiere_texte(df_main['Aussage'].tolist())

    print("Lade Sprachmodell und kodiere bereinigte Aktionen in Vektoren...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_main['aktion_lemmatisiert'].tolist(), show_progress_bar=True)

    # --- STUFE 2: AUTOMATISCHE PARAMETER-OPTIMIERUNG ---
    print("\n--- STUFE 2: Suche nach optimalen Parametern für granulare Cluster ---")
    umap_nachbarn_liste = [10, 15, 25]
    hdbscan_cluster_groesse_liste = list(range(10, 60, 2))
    parameter_grid = list(itertools.product(umap_nachbarn_liste, hdbscan_cluster_groesse_liste))
    optimierungs_ergebnisse = []
    # Schleife über alle Parameter-Kombinationen.
    for n_nachbarn, min_cluster_groesse in parameter_grid:
        reducer = umap.UMAP(n_neighbors=n_nachbarn, n_components=5, metric='cosine', random_state=42)
        umap_embeddings_test = reducer.fit_transform(embeddings)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_groesse, metric='euclidean', gen_min_span_tree=True)
        clusterer.fit(umap_embeddings_test)
        validity_score = clusterer.relative_validity_
        optimierungs_ergebnisse.append({'n_nachbarn': n_nachbarn, 'min_cluster_groesse': min_cluster_groesse, 'dbcv_score': validity_score})
    
    results_df = pd.DataFrame(optimierungs_ergebnisse)
    beste_parameter = results_df.loc[results_df['dbcv_score'].idxmax()]
    print("\nBeste gefundene Parameter-Kombination:\n", beste_parameter)
    
    # VISUELLE BEGRÜNDUNG SPEICHERN
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    sns.lineplot(data=results_df, x='min_cluster_groesse', y='dbcv_score', hue='n_nachbarn', ax=ax, marker='o', palette='viridis').set_title('DBCV Score vs. Parameter (höher ist besser)')
    plt.suptitle('Visuelle Begründung der Parameterwahl', fontsize=16)
    fig_path = os.path.join(speicherordner, "A_parameter_optimierung_analyse.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Analyse-Plots gespeichert unter: {fig_path}")

    # --- STUFE 3: FINALES GRANULARES CLUSTERING ---
    print("\n--- STUFE 3: Führe finales granulares Clustering durch ---")
    final_n_nachbarn = int(beste_parameter['n_nachbarn'])
    final_min_cluster_groesse = int(beste_parameter['min_cluster_groesse'])
    final_reducer = umap.UMAP(n_neighbors=final_n_nachbarn, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
    final_umap_embeddings = final_reducer.fit_transform(embeddings)
    final_clusterer = hdbscan.HDBSCAN(min_cluster_size=final_min_cluster_groesse, metric='euclidean', allow_single_cluster=True)
    final_clusterer.fit(final_umap_embeddings)
    df_main['cluster_id'] = final_clusterer.labels_
    
    # --- STUFE 4: ALGORITHMISCHE BENENNUNG DER CLUSTER ---
    print("\n--- STUFE 4: Benenne granulare Cluster mit Keywords ---")
    granular_keyword_names = _name_clusters_with_keywords(df_main, 'cluster_id', 'aktion_lemmatisiert', anzahl_keywords=8)
    df_main['Cluster'] = df_main['cluster_id'].map(granular_keyword_names)
    # Temporärer Name für Rauschen
    df_main.loc[df_main['cluster_id'] == -1, 'Cluster'] = "RAUSCHEN_PLATZHALTER"
    
    # --- STUFE 5: KLASSIFIZIERUNG DES RAUSCHENS MIT GEMINI ---
    print("\n--- STUFE 5: Klassifiziere 'Nicht eindeutige' Aussagen mit der Gemini API ---")
    # Schleife über alle Zeilen, die als Rauschen markiert sind.
    for index, row in df_main[df_main['Cluster'] == "RAUSCHEN_PLATZHALTER"].iterrows():
        print(f"  Prüfe Aussage: '{row['Aussage'][:60]}...'")
        prompt = CLASSIFY_NOISE_PROMPT.format(statement=row['Aussage'])
        classification = _get_api_response(prompt)
        
        if "Konkrete Massnahme" in classification:
            df_main.loc[index, 'Cluster'] = "Sonstiges"
        else:
            df_main.loc[index, 'Cluster'] = "Keine Maßnahme"
        print(f"    -> Klassifiziert als: {df_main.loc[index, 'Cluster']}")
        time.sleep(1) 

    # --- STUFE 6: ERSTELLUNG DER ÜBERKATEGORIEN MIT GEMINI API ---
    print("\n--- STUFE 6: Erstelle Überkategorien mit der Gemini API ---")
    # Sammle alle finalen, granularen Cluster-Namen (außer den speziellen Kategorien)
    granulare_namen_liste = df_main[~df_main['Cluster'].isin(["Nicht eindeutig", "Sonstiges", "Keine Maßnahme"])]['Cluster'].unique().tolist()
    
    if granulare_namen_liste:
        formatted_category_list = json.dumps(granulare_namen_liste, indent=2)
        prompt = GROUP_CATEGORIES_PROMPT.format(
            num_categories=len(granulare_namen_liste),
            category_list=formatted_category_list
        )
        json_response_str = _get_api_response(prompt)
        
        try:
            mapping_dict = json.loads(json_response_str)
            df_main['Überkategorie'] = df_main['Cluster'].map(mapping_dict)
            # Fülle die speziellen Kategorien und eventuelle Fehler auf
            df_main.loc[df_main['Cluster'] == 'Sonstiges', 'Überkategorie'] = 'Sonstige Maßnahmen'
            df_main.loc[df_main['Cluster'] == 'Keine Maßnahme', 'Überkategorie'] = 'Allgemeine Aussagen'
            df_main['Überkategorie'].fillna("Unkategorisiert", inplace=True)
            
            unique_super_cats = df_main[~df_main['Überkategorie'].isin(["Unkategorisiert", "Allgemeine Aussagen"])]['Überkategorie'].nunique()
            print(f"  {unique_super_cats} einzigartige Überkategorien wurden von der API erstellt.")
        except json.JSONDecodeError:
            print("  Fehler: Konnte die JSON-Antwort der API nicht parsen. Überkategorien werden nicht zugewiesen.")
            df_main['Überkategorie'] = "Fehler bei Kategorisierung"
    else:
        print("  Keine granularen Namen zum Gruppieren gefunden.")
        df_main['Überkategorie'] = "N/A"
        
    # --- STUFE 7: FINALES SPEICHERN ---
    print("\n--- STUFE 7: Speichere finalen Report ---")
    
    df_final_output = df_main[['Unternehmen', 'Typ', 'Aussage', 'Status', 'Cluster', 'Überkategorie']]
    
    final_path = os.path.join(speicherordner, "clusters_final.xlsx")
    df_final_output.to_excel(final_path, index=False)
    print(f"\nAnalyse vollständig abgeschlossen. Report gespeichert unter: '{final_path}'")
