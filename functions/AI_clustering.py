import os
import json
import pandas as pd
import time
from dotenv import load_dotenv
import google.generativeai as genai
import re
from tqdm import tqdm


load_dotenv()

# --- VORDEFINIERTE KATEGORIEN UND PROMPTS ---
PREDEFINED_CATEGORIES = [
    "Collaborations & Partnerships",
    "Education & Training & Awareness",
    "Research",
    "Changes in procurement",
    "Governance & Strategy & Plans",
    "Monitoring & Assessment",
    "Financial Actions & Investments", 
    "Protecting existing Animals & Wildlife",
    "Creating new Animals & Wildlife",
    "Protecting existing Trees & Plants",
    "Creating new Trees & Plants",
    "Water & Coast & Ocean",
    "Landuse and Agriculture", 
    "Pollution Control", 
    "Reduction in resource consumption",
    "Framework Alignment: CSRD / ESRS",
    "Framework Alignment: GRI",
    "Framework Alignment: TNFD",
    "Framework Alignment: SBTN",
    "No Biodiversity Relevance",
    "General statement"
]


CLASSIFICATION_PROMPT = """
You are a balanced but precise classification engine. Your task is to classify corporate statements by following a hierarchical logic. Your goal is to accurately identify statements with a substantive, direct link to biodiversity.

**Step 1: Default Assumption & Core Analysis**
- Your default assumption is that the statement's category is **"No Biodiversity Relevance"**.
- First, identify the core action or subject of the statement.

**Step 2: The Explicit Link Gate**
- Next, determine if the statement establishes a **direct and explicit link** between its core action and a tangible biodiversity outcome (related to ecosystems, habitats, or species).
- **Ask this key question:** Is the biodiversity benefit a core, stated purpose of the action, or is it merely an indirect, potential side-effect? A simple mention of "environment" or "sustainability" is not enough.
- **The action must be the *cause*, and the biodiversity outcome must be the *direct effect*.**

- **Exclusion List:** Statements focused primarily on the following topics are **NOT** relevant, UNLESS they explicitly describe how this action directly leads to a specific habitat or species outcome:
  - General reduction of CO2/GHG emissions or climate action.
  - Improving energy efficiency or switching to renewables.
  - General reduction of water usage.
  - General waste reduction, recycling, or circular economy initiatives.

- **Example of failure:** "We are reducing our water consumption to protect the local environment." -> Fails. The link is not explicit. What is the specific biodiversity outcome?
- **Example of success:** "We are reducing our water extraction from the Silver Creek by 30% to maintain the required water levels for the native fish population's spawning season." -> Passes. The action (water reduction) is explicitly and directly linked to a specific species outcome.

- **If no direct and explicit link to a biodiversity outcome is described, your final and only answer is "No Biodiversity Relevance". Stop here.**

**Step 3: Provisional Classification**
- **If the statement passes the gate, its category is now provisionally "General Statement".**

**Step 4: Specific Category Refinement**
- Now, review the other categories.
- To move the statement from "General Statement" to a more specific category, there must be a **clear and concrete description of a specific action**. A vague intention is not sufficient.
- If you are unsure, or if the statement remains a high-level commitment, the classification defaults to **"General Statement"**.

**Crucially, respond *only* with the category name itself.** Do not add any explanation, introduction, or quotation marks.

---
**Predefined Categories:**
{category_list}

**Statement to classify:**
"{statement}"

**Category:**
"""

STATUS_PROMPT = """
Analyze the following corporate statement.
Is it describing a future goal, a plan, or an intention? Or is it describing an action that has already been implemented or is currently in progress?
- If it is a plan, goal, or intention for the future (e.g., "we will," "we aim to," "our goal is"), respond with the single word: **planned**
- If it is a completed or ongoing action (e.g., "we have," "we did," "we are"), respond with the single word: **done**

Respond only with "planned" or "done".

**Statement:**
"{statement}"

**Status:**
"""

METRIC_PROMPT = """
You are a specialized analyst for sustainability reporting frameworks. Your task is to determine if a corporate statement explicitly mentions a metric or standard from "CSRD / ESRS", "GRI", "TNFD", or "SBTN". Follow these steps precisely:

1.  **Direct Framework Mention:** First, scan the statement for a direct mention of the framework names or their common identifiers (e.g., "CSRD", "ESRS E4", "GRI 304", "TNFD", "SBTN").
    * If "CSRD" or "ESRS" is mentioned, respond with **CSRD / ESRS**. Stop.
    * If "GRI" is mentioned, respond with **GRI**. Stop.
    * If "TNFD" is mentioned, respond with **TNFD**. Stop.
    * If "SBTN" is mentioned, respond with **SBTN**. Stop.

2.  **Specific Metric Mention:** If no framework is named, check if the statement describes a specific, quantifiable metric that is characteristic of these frameworks (e.g., area of restored habitat in hectares, number of IUCN Red List species affected, financial value of nature-related risks).
    * If such a specific metric is mentioned, respond with **other**. Stop.

3.  **Default:** If neither a framework name nor a specific, quantifiable metric is mentioned, respond with **no**.

**Your Final Output:**
Respond *only* with one of the exact following values: **CSRD / ESRS**, **GRI**, **TNFD**, **SBTN**, **other**, **no**. Do not add any other text.

**Statement to analyze:**
"{statement}"

**Framework Metric:**
"""

# --- Hilfsfunktionen ---
# Funktion zum Extrahieren aller Einträge aus den JSON-Dateien
def _extrahiere_alle_eintraege(input_ordner: str) -> list[dict]:
    alle_eintraege = []
    actions_key = "actions"
    metrics_key = "metrics"
    print(f"Extrahiere Daten aus '{actions_key}' und '{metrics_key}'.")

    dateien = os.listdir(input_ordner)
    print(f"INFO: {len(dateien)} Dateien/Ordner im Input-Verzeichnis gefunden.")

    # Schleife über alle Dateien im Input-Ordner
    for dateiname in dateien:
        if not dateiname.lower().endswith(".json"):
            continue
        
        print(f"  - Verarbeite Datei: {dateiname}")
        unternehmen = os.path.splitext(dateiname)[0]
        voller_pfad = os.path.join(input_ordner, dateiname)
        
        eintraege_pro_datei = 0
        try:
            with open(voller_pfad, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Schleife über alle gefundenen Passagen in einer Datei
            for passage_block in data.get("biodiversity_passages", []):
                keywords_str = ", ".join(passage_block.get("found_keywords", []))
                
                actions_in_passage = passage_block.get(actions_key, [])
                metrics_in_passage = passage_block.get(metrics_key, [])
                eintraege_pro_datei += len(actions_in_passage) + len(metrics_in_passage)
                
                # Schleife über alle "actions" in einer Passage
                for action_string in actions_in_passage:
                    alle_eintraege.append({"Unternehmen": unternehmen, "Typ": "Action", "Aussage": action_string.strip("'\""), "Keywords": keywords_str})
                
                # Schleife über alle "metrics" in einer Passage
                for metric_string in metrics_in_passage:
                    alle_eintraege.append({"Unternehmen": unternehmen, "Typ": "Metric", "Aussage": metric_string.strip("'\""), "Keywords": keywords_str})
            
            print(f"    - {eintraege_pro_datei} Einträge aus dieser Datei extrahiert.")

        except Exception as e:
            print(f"Fehler beim Lesen von {dateiname}: {e}")
            
    print(f"Extraktion abgeschlossen. Insgesamt {len(alle_eintraege)} Einträge gefunden.")
    return alle_eintraege

# Generalisierte Funktion für API-Aufrufe mit Wiederholungslogik
def _get_api_response(prompt_template: str, statement: str, fallback: str, retries: int = 3, delay: int = 5) -> str:
    for attempt in range(retries):
        try:
            if '{category_list}' in prompt_template:
                prompt = prompt_template.format(category_list="\n".join(f"- {c}" for c in PREDEFINED_CATEGORIES), statement=statement)
            else:
                prompt = prompt_template.format(statement=statement)

            model = genai.GenerativeModel("gemini-2.5-flash-lite") 
            response = model.generate_content(prompt)
            return response.text.strip() 

        except Exception as e:
            # Prüft, ob es der letzte Versuch war
            if attempt < retries - 1:
                wait_time = delay * (2 ** attempt)  # Exponential backoff: 5s, 10s, 20s
                print(f"    Fehler bei API-Aufruf (Versuch {attempt + 1}/{retries}): {e}. Warte {wait_time}s...")
                time.sleep(wait_time)
            else:
                error_message = f"Finaler Fehler nach {retries} Versuchen: {e}"
                print(f"    {error_message}")
               
                return fallback 
            
    return fallback # Sollte nie erreicht werden, aber als Absicherung

def create_robust_merge_key(name: str) -> str:
    if not isinstance(name, str): return ""
    name = name.lower()
    
    # Ersetzt Unterstriche und Bindestriche durch Leerzeichen
    name = name.replace('_', ' ').replace('-', ' ')
    
    # Erweiterte Liste von Suffixen und generischen Begriffen
    words_to_remove = [
        # Skandinavien
        'ab', 'asa', 
        # UK / USA / International
        'plc', 'limited', 'ltd', 'inc', 'incorporated', 'corp', 'corporation', 'group',
        # Deutschland / Österreich
        'ag', 'gmbh', 'se',
        # Frankreich / Spanien / Italien / Lateinamerika
        'sa', 'srl', 'spa',
        # Niederlande / Belgien
        'nv', 'bv',
        # Allgemein
        'the', 'holding', 'holdings', 
        # Report-spezifisch
        'sustainability', 'report', 'relevant', 'passages', 'annual', 'integrated'
    ]
    
    pattern = r'\b(' + '|'.join(words_to_remove) + r')\b'
    name = re.sub(pattern, '', name)
    name = re.sub(r'(_)?\d{4}', '', name)
    name = re.sub(r'[^a-z0-9]', '', name)
    return name.strip()

# Führt die vollständige KI-Analyse mit Checkpoint- und Resume-Funktion durch
def fuehre_top_down_klassifizierung_durch(input_ordner: str, summary_excel_path: str, output_ordner: str):
    print("--- Beginne Top-Down-Analyse ---")
    
    classification_output_ordner = os.path.join(output_ordner, "Top_Down_Analyse")
    os.makedirs(classification_output_ordner, exist_ok=True)
    
    # Pfad für die Zwischenspeicherung 
    checkpoint_path = os.path.join(classification_output_ordner, "checkpoint_report.xlsx")
    
    # Alle zu verarbeitenden Einträge laden
    alle_eintraege = _extrahiere_alle_eintraege(input_ordner)
    if not alle_eintraege:
        print("Keine Aktionen oder Metriken gefunden.")
        return
    df_full = pd.DataFrame(alle_eintraege)

    # Zum testen aktivieren:
    #df_full = df_full[df_full.index % 90 == 0]


    # DataFrame für Ergebnisse initialisieren: Entweder aus Checkpoint laden oder neu erstellen
    if os.path.exists(checkpoint_path):
        print(f"Lade Fortschritt aus Checkpoint-Datei: {checkpoint_path}")
        df_results = pd.read_excel(checkpoint_path)
    else:
        print("Keine Checkpoint-Datei gefunden. Starte eine neue Analyse.")
        df_results = pd.DataFrame()

    # Bestimmen, welche Aussagen noch verarbeitet werden müssen
    if not df_results.empty:
        processed_statements = set(df_results['Aussage'])
        df_todo = df_full[~df_full['Aussage'].isin(processed_statements)].copy()
    else:
        df_todo = df_full.copy()

    if df_todo.empty:
        print("Alle Aussagen wurden bereits verarbeitet.")
    else:
        print(f"\nVerarbeite {len(df_todo)} verbleibende von insgesamt {len(df_full)} Aussagen...")
        
        neue_ergebnisse = []

        # Schleife über die noch zu verarbeitenden Aussagen
        for index, row in tqdm(df_todo.iterrows(), total=df_todo.shape[0], desc="Verarbeite Aussagen"):
            aussage = row['Aussage']
            
            # API-Aufrufe
            kategorie = _get_api_response(CLASSIFICATION_PROMPT, aussage, fallback="API Fehler")
            time.sleep(1)
            status = _get_api_response(STATUS_PROMPT, aussage, fallback="API Fehler")
            time.sleep(1)
            metrik = _get_api_response(METRIC_PROMPT, aussage, fallback="API Fehler")
            time.sleep(1)

            # Neue Zeile für das Ergebnis-DataFrame erstellen
            new_row = row.to_dict()
            new_row['Kategorie'] = kategorie
            new_row['Status'] = status
            new_row['Metric'] = metrik
            neue_ergebnisse.append(new_row)

            # Nach jedem Eintrag den Fortschritt speichern
            temp_df_to_save = pd.DataFrame(neue_ergebnisse)
            df_to_save = pd.concat([df_results, temp_df_to_save], ignore_index=True)
            df_to_save.to_excel(checkpoint_path, index=False)
        
        # Das finale Ergebnis-DataFrame nach der Schleife aktualisieren
        df_results = pd.concat([df_results, pd.DataFrame(neue_ergebnisse)], ignore_index=True)

    print("Alle Aussagen erfolgreich verarbeitet.")

    # Anreicherung mit Metadaten
    print("\nReichere Report mit Metadaten an...")
    df_enriched = df_results.copy()
    try:
        df_summary = pd.read_excel(summary_excel_path)
        columns_to_merge = ['Filename', 'Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']
        df_summary_subset = df_summary[columns_to_merge].copy()
        
        # Schlüssel für beide DataFrames erstellen
        print("Erstelle robuste Merge-Schlüssel...")
        df_enriched['merge_key'] = df_enriched['Unternehmen'].apply(create_robust_merge_key)
        df_summary_subset['merge_key'] = df_summary_subset['Company'].apply(create_robust_merge_key)

        # Doppelte Schlüssel in der Metadaten-Tabelle entfernen
        df_summary_subset.drop_duplicates(subset=['merge_key'], keep='first', inplace=True)

        # Merge durchführen
        print("Führe Merge der Tabellen durch...")
        df_enriched = pd.merge(df_enriched, df_summary_subset, on='merge_key', how='left')
        # Schleife zum Auffüllen fehlender Werte
        for col in columns_to_merge:
            if col not in df_enriched:
                df_enriched[col] = 'N/A'
            else:
                if col == 'Company':
                    # Fülle nur dort auf, wo der Merge erfolgreich war
                    # Die Original-Company aus dem Summary-Sheet wird behalten
                    pass
                df_enriched[col].fillna('N/A', inplace=True)

    except FileNotFoundError as e:
        print(f"Warnung: Metadaten-Excel '{summary_excel_path}' nicht gefunden oder fehlerhaft. ({e})")
        # Schleife zum Hinzufügen von leeren Spalten, falls die Datei nicht existiert
        for col in ['Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']:
            df_enriched[col] = 'N/A'

    #Entferne alle Zeilen mit der Kategorie "No Biodiversity Relevance"
    print("\nFiltere irrelevante Aussagen heraus...")
    rows_before = len(df_enriched)
    
    df_final = df_enriched[df_enriched['Kategorie'] != 'No Biodiversity Relevance'].copy()
    
    rows_after = len(df_final)
    print(f"-> {rows_before - rows_after} Aussagen der Kategorie 'No Biodiversity Relevance' entfernt.")
    print(f"-> {rows_after} relevante Aussagen verbleiben im finalen Report.")

    # Speichern des finalen Reports
    final_path = os.path.join(classification_output_ordner, "top_down_klassifizierungs_report.xlsx")
    output_columns = ['Unternehmen', 'Typ', 'Aussage', 'Status', 'Kategorie', 'Metric', 'Keywords', 'Company', 'Country', 'Rating', 'Primary Listing', 'Industry Classification']
    df_final.to_excel(final_path, index=False, columns=output_columns)
    
    print(f"\n--- Analyse vollständig abgeschlossen. ---\nFinaler Report gespeichert unter: '{final_path}'")
   