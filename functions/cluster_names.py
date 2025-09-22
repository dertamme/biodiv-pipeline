# import json
# import os
# import time
# from dotenv import load_dotenv
# import pandas as pd
# from functions.prompts import gemini_model_version, NAME_CLUSTER_PROMPT
# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import umap
# import hdbscan
# import numpy as np

# # Laden der Umgebungsvariablen (z.B. GOOGLE_API_KEY)
# load_dotenv()

# # Prompt, der die KI die Anzahl der Überkategorien selbst finden lässt
# GROUP_CATEGORIES_PROMPT = """
# You are a data analyst. Based on the following list of {num_categories} granular action categories, group them into a meaningful number of high-level super-categories. The number of super-categories should be based on the thematic coherence of the items. Try to avoid large overcategories with more than 5 categories, unless it really makes sense.

# Respond with a single JSON object. The keys of the object must be the original granular category names from the list, and the value for each key must be the assigned super-category name. Ensure every single category from the list is a key in your JSON response.

# Example Response Format:
# {{
#   "Green Roof Installation": "Infrastructure & Operations",
#   "FSC Certified Sourcing": "Supply Chain Management",
#   "Biodiversity Impact Reporting": "Strategy & Reporting"
# }}

# Here is the list of granular categories to group:
# {category_list}

# JSON Response:
# """

# # Prompt, der die KI anweist, die Zuordnung zu überprüfen
# CHALLENGE_SUPER_CATEGORY_PROMPT = """
# You are a critical data analyst. The super-category '{super_category_name}' was created to group the following granular categories.
# Review the list and identify any granular categories that DO NOT fit well thematically. If all categories fit, that's okay. 

# Respond with a single JSON object with one key: "misplaced_categories". The value should be a list of the names that should be moved. If all categories fit well, the list should be empty.

# Example Response:
# {{
#   "misplaced_categories": ["Tree Planting Initiatives", "Community Engagement Programs"]
# }}

# Super-Category: '{super_category_name}'
# List of Granular Categories:
# {category_list}

# JSON Response:
# """

# # NEUER Prompt, der die KI anweist, eine Kategorie neu zuzuordnen
# REASSIGN_CATEGORY_PROMPT = """
# You are a data analyst. The granular category '{granular_name}' was identified as not belonging to its original super-category.

# Please choose the best-fitting super-category for it from the following list of existing, valid super-categories.
# Respond *only* with the name of the chosen super-category from the list. Do not invent a new one.

# List of valid Super-Categories:
# {super_category_list}

# Granular Category to re-assign: '{granular_name}'

# Best-fitting Super-Category:
# """

# # Gemini Modell Konfiguration
# gemini_model_version = "gemini-1.5-flash-latest"


# # --- HELFERFUNKTIONEN ---

# def _get_single_name_from_api(prompt_template: str, examples: list[str]) -> str:
#     """Fragt die API nach einem einzelnen Namen für eine Gruppe von Beispielen."""
#     try:
#         # Schleife über die Beispiele, um den Prompt zu formatieren.
#         beispiele_formatiert = "\n".join([f"- {text}" for text in examples])
#         prompt = prompt_template.format(examples=beispiele_formatiert)

#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key: return "FEHLER: GOOGLE_API_KEY nicht gefunden."
            
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel(gemini_model_version)
#         response = model.generate_content(prompt)
        
#         return response.text.strip().replace('"', '')
#     except Exception as e:
#         print(f"    Fehler bei API-Aufruf: {e}")
#         return "FEHLER bei API-Aufruf"

# def _get_super_category_mapping_from_api(category_list: list[str]) -> dict:
#     """Fragt die API nach einer Gruppierung für eine ganze Liste von Kategorien."""
#     try:
#         formatted_category_list = json.dumps(category_list, indent=2)
#         prompt = GROUP_CATEGORIES_PROMPT.format(
#             num_categories=len(category_list),
#             category_list=formatted_category_list
#         )

#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key: return {"error": "GOOGLE_API_KEY nicht gefunden."}
            
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel(gemini_model_version)
#         response = model.generate_content(prompt)
        
#         cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
#         mapping = json.loads(cleaned_response)
#         return mapping
#     except Exception as e:
#         print(f"    Fehler beim Parsen der Super-Kategorie-Antwort: {e}")
#         return {}

# def _challenge_super_category_assignment(super_category_name: str, category_list: list[str]) -> list:
#     """Fragt die API, welche Kategorien nicht in eine Überkategorie passen."""
#     try:
#         formatted_category_list = json.dumps(category_list, indent=2)
#         prompt = CHALLENGE_SUPER_CATEGORY_PROMPT.format(
#             super_category_name=super_category_name,
#             category_list=formatted_category_list
#         )
#         api_key = os.getenv("GOOGLE_API_KEY")
#         if not api_key: return ["FEHLER: GOOGLE_API_KEY nicht gefunden."]

#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel(gemini_model_version)
#         response = model.generate_content(prompt)

#         cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
#         result = json.loads(cleaned_response)
#         return result.get("misplaced_categories", [])
#     except Exception as e:
#         print(f"    Fehler bei der Überprüfung der Zuordnung: {e}")
#         return []

# def _get_corrected_super_category(granular_name: str, super_category_list: list[str]) -> str:
#     """NEUE Funktion: Fragt die API nach der korrekten Überkategorie für ein fehlplatziertes Thema."""
#     try:
#         # Schleife über die Beispiele, um den Prompt zu formatieren.
#         formatted_super_category_list = "\n".join([f"- {name}" for name in super_category_list])
#         prompt = REASSIGN_CATEGORY_PROMPT.format(
#             granular_name=granular_name,
#             super_category_list=formatted_super_category_list
#         )
#         return _get_single_name_from_api(prompt, []) # Nutzt die allgemeine API-Funktion mit einem leeren Beispiel-Array
#     except Exception as e:
#         print(f"    Fehler bei der Neuzuordnung: {e}")
#         return "Neuzuordnung fehlgeschlagen"


# # --- HAUPTFUNKTION ---

# def name_and_categorize_clusters(ordner: str) -> None:
#     """
#     Führt einen vollständigen, mehrstufigen Benennungs- und Kategorisierungsprozess durch.
#     """
#     print(f"--- Starte hierarchische Benennung und Kategorisierung im Ordner: {ordner} ---")

#     try:
#         alle_dateien_im_ordner = os.listdir(ordner)
#     except FileNotFoundError:
#         print(f"FEHLER: Der angegebene Ordner '{ordner}' wurde nicht gefunden.")
#         return

#     # Schleife über alle zu verarbeitenden CSV-Dateien im Ordner.
#     for dateiname in alle_dateien_im_ordner:
        
#         # Schleife über die Spalten, um die Cluster-ID-Spalte zu finden.
#         if not dateiname.endswith(".csv"): continue
#         voller_pfad = os.path.join(ordner, dateiname)
#         try:
#             df_head = pd.read_csv(voller_pfad, nrows=0)
#             hat_cluster_spalte = 'cluster_id' in df_head.columns or 'sub_cluster_id' in df_head.columns
#             ist_bereits_verarbeitet = 'granular_cluster_name' in df_head.columns or 'super_category_name' in df_head.columns
#             if not hat_cluster_spalte or ist_bereits_verarbeitet: continue
#         except Exception: continue

#         print(f"\n--- STUFE 1: Benenne granulare Cluster in Datei: {dateiname} ---")
#         df = pd.read_csv(voller_pfad)

#         cluster_spalte = 'cluster_id' if 'cluster_id' in df.columns else 'sub_cluster_id'

#         df['granular_cluster_name'] = ""
#         einzigartige_cluster = sorted(df[df[cluster_spalte] != -1][cluster_spalte].unique())

#         # Schleife über jeden granularen Cluster, um ihn zu benennen.
#         for cluster_id in einzigartige_cluster:
#             print(f"    Benenne granularen Cluster {cluster_id}...")
#             stichprobe = df[df[cluster_spalte] == cluster_id]['aktion'].head(300).tolist()
#             cluster_name = _get_single_name_from_api(NAME_CLUSTER_PROMPT, stichprobe)
#             print(f"      -> Erhaltener Name: '{cluster_name}'")
#             df.loc[df[cluster_spalte] == cluster_id, 'granular_cluster_name'] = cluster_name
#             time.sleep(2)
        
#         df.loc[df[cluster_spalte] == -1, 'granular_cluster_name'] = "Rauschen (Noise)"

#         # --- STUFE 2: Erstelle erste Überkategorien mit der Gemini API ---
#         print("\n--- STUFE 2: Erstelle erste Überkategorien mit der Gemini API ---")
        
#         benannte_cluster = df[df['granular_cluster_name'] != "Rauschen (Noise)"]['granular_cluster_name'].unique().tolist()
        
#         if len(benannte_cluster) < 5:
#             print("    Nicht genügend unterschiedliche granulare Cluster für die Bildung von Überkategorien.")
#             df['super_category_name'] = "N/A"
#         else:
#             print(f"    Sende {len(benannte_cluster)} granulare Kategorienamen an die API zur Gruppierung...")
#             mapping_dict = _get_super_category_mapping_from_api(benannte_cluster)
            
#             if mapping_dict:
#                 df['super_category_name'] = df['granular_cluster_name'].map(mapping_dict)
#                 df['super_category_name'].fillna("Unkategorisiert", inplace=True)
#             else:
#                 print("    Konnte keine Überkategorien von der API erhalten.")
#                 df['super_category_name'] = "Fehler bei Kategorisierung"

#             # --- STUFE 3: Überprüfung und Neuzuordnung der Zuordnung ---
#             print("\n--- STUFE 3: Überprüfe und korrigiere die Zuordnung der Überkategorien ---")
#             all_misplaced = []
#             # Schleife über jede erstellte Überkategorie.
#             super_categories_in_df = df[df['super_category_name'].notna() & ~df['super_category_name'].isin(["Unkategorisiert", "Fehler bei Kategorisierung"])]['super_category_name'].unique()
#             for super_cat_name in super_categories_in_df:
#                 print(f"    Überprüfe Überkategorie: '{super_cat_name}'")
#                 zugeordnete_namen = df[df['super_category_name'] == super_cat_name]['granular_cluster_name'].unique().tolist()
                
#                 misplaced_names = _challenge_super_category_assignment(super_cat_name, zugeordnete_namen)
#                 if misplaced_names:
#                     print(f"      -> Unpassende Themen gefunden: {misplaced_names}")
#                     all_misplaced.extend(misplaced_names)
#                 else:
#                     print("      -> Alle Themen passen.")
#                 time.sleep(2)

#             # Weise die unpassenden Themen einer neuen Kategorie zu
#             if all_misplaced:
#                 print(f"\n--- STUFE 3.5: Ordne {len(all_misplaced)} unpassende Themen neu zu ---")
#                 # Erstelle eine Liste der validen Überkategorien, die nach der Überprüfung noch existieren.
#                 valid_super_categories = df[~df['granular_cluster_name'].isin(all_misplaced)]['super_category_name'].unique().tolist()
#                 valid_super_categories = [name for name in valid_super_categories if name not in ["Unkategorisiert", "Fehler bei Kategorisierung"]]

#                 if not valid_super_categories:
#                     print("      -> Keine validen Überkategorien mehr übrig. Unpassende Themen werden als 'Sonstige' markiert.")
#                     df.loc[df['granular_cluster_name'].isin(all_misplaced), 'super_category_name'] = "Sonstige Themen (Überprüft)"
#                 else:
#                     # Schleife über jedes unpassende Thema und ordne es neu zu.
#                     for misplaced_name in all_misplaced:
#                         print(f"    Ordne '{misplaced_name}' neu zu...")
#                         corrected_super_cat = _get_corrected_super_category(misplaced_name, valid_super_categories)
#                         print(f"      -> Neue Überkategorie: '{corrected_super_cat}'")
#                         df.loc[df['granular_cluster_name'] == misplaced_name, 'super_category_name'] = corrected_super_cat
#                         time.sleep(2)

#         # --- STUFE 4: Speichern des finalen Ergebnisses ---
#         try:
#             basisname_ohne_ext = os.path.splitext(dateiname)[0]
#             neuer_dateiname = f"{basisname_ohne_ext}_final_hierarchie.csv"
#             output_pfad = os.path.join(ordner, neuer_dateiname)
            
#             finale_spalten = ['aktion', 'cluster_id', 'granular_cluster_name', 'super_category_name']
#             df[finale_spalten].to_csv(output_pfad, index=False, encoding='utf-8-sig')
#             print(f"\n  Finale Hierarchie erfolgreich gespeichert in: '{output_pfad}'")
#         except Exception as e:
#             print(f"    Fehler beim Speichern der finalen Datei '{neuer_dateiname}': {e}")
    
#     print("\n--- Prozess vollständig abgeschlossen ---")
