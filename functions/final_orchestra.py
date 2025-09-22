import os
import pandas as pd
import json


def erstelle_finalen_report(json_input_ordner: str, cluster_ergebnis_xlsx_pfad: str, output_ordner: str):
    """
    Sammelt alle Aktionen/Metriken aus den JSON-Dateien und reichert sie mit den
    Cluster- und Überkategorie-Informationen aus der Analyse-Excel-Datei an.

    Args:
        json_input_ordner (str): Der Ordner, der die originalen JSON-Dateien mit den 
                                 'summarized_actions'/'summarized_metrics' enthält.
        cluster_ergebnis_xlsx_pfad (str): Der Pfad zur finalen Excel-Datei, die die 
                                          Cluster-Zuordnungen enthält.
        output_ordner (str): Der Ordner, in dem der finale Report gespeichert wird.
    """
    print("--- Starte Erstellung des finalen Reports ---")
    
    # --- Stufe 1: Lade das Cluster-Ergebnis als schnelles Nachschlagewerk ---
    try:
        print(f"Lese Cluster-Ergebnis-Excel von: {cluster_ergebnis_xlsx_pfad}")
        df_lookup = pd.read_excel(cluster_ergebnis_xlsx_pfad)
        
        df_lookup.drop_duplicates(subset=['Aussage'], keep='first', inplace=True)
        df_lookup.set_index('Aussage', inplace=True)
        print("Cluster-Ergebnisse erfolgreich geladen und für die Suche vorbereitet.")
    except FileNotFoundError:
        print(f"FEHLER: Die Cluster-Ergebnis-Datei wurde nicht gefunden: {cluster_ergebnis_xlsx_pfad}")
        return
    except Exception as e:
        print(f"Ein unerwarteter Fehler beim Laden der Cluster-Excel-Datei ist aufgetreten: {e}")
        return


    report_daten = []

    # --- Stufe 2: Iteriere durch die originalen JSON-Dateien ---
    print(f"Durchsuche JSON-Dateien in: {json_input_ordner}")
    
    # Schleife über alle Dateien im Input-Ordner.
    for dateiname in os.listdir(json_input_ordner):
        if not dateiname.lower().endswith(".json"):
            continue

        unternehmen = os.path.splitext(dateiname)[0]
        voller_pfad = os.path.join(json_input_ordner, dateiname)

        try:
            with open(voller_pfad, 'r', encoding='utf-8-sig') as f: 
                data = json.load(f)
            
            # Schleife über alle Passage-Blöcke in einer JSON-Datei.
            for passage_block in data.get("biodiversity_passages", []):
                
                # Schleife über alle Aktionen im Block.
                for action_string in passage_block.get("summarized_actions", []):
                    try:
                        if ":" in action_string:
                            status, text = action_string.split(":", 1)
                            status = status.strip().capitalize()
                        else:
                            status = "Unknown"
                            text = action_string
                        
                        reiner_text = text.strip().strip("'\"")
                        
                        cluster_info = df_lookup.loc[reiner_text]
                        
                        report_daten.append({
                            "Unternehmen": unternehmen,
                            "Typ": "Action",
                            "Aussage": reiner_text,
                            "Status": status,
                            "Cluster": cluster_info.get('Cluster', 'N/A'),
                            "Überkategorie": cluster_info.get('Überkategorie', 'N/A')
                        })
                    except KeyError:
                        # Ignoriere, falls eine Aktion nicht gefunden wird
                        continue
                    except Exception as e:
                        print(f"  Fehler beim Parsen der Zeile '{action_string}': {e}")
                        continue

                # Schleife über alle Metriken im Block.
                for metric_string in passage_block.get("summarized_metrics", []):
                    try:
                        if ":" in metric_string:
                            status, text = metric_string.split(":", 1)
                            status = status.strip().capitalize()
                        else:
                            status = "Unknown"
                            text = metric_string

                        reiner_text = text.strip().strip("'\"")
                        
                        cluster_info = df_lookup.loc[reiner_text]
                        
                        report_daten.append({
                            "Unternehmen": unternehmen,
                            "Typ": "Metric",
                            "Aussage": reiner_text,
                            "Status": status,
                            "Cluster": cluster_info.get('Cluster', 'N/A'),
                            "Überkategorie": cluster_info.get('Überkategorie', 'N/A')
                        })
                    except KeyError:
                        # Ignoriere, falls eine Metrik nicht gefunden wird
                        continue
                    except Exception as e:
                        print(f"  Fehler beim Parsen der Zeile '{metric_string}': {e}")
                        continue

        except Exception as e:
            print(f"Fehler beim Verarbeiten der Datei {dateiname}: {e}")

    # --- Stufe 3: Erstelle und speichere den finalen Report ---
    if not report_daten:
        print("Keine passenden Aktionen oder Metriken gefunden, um einen Report zu erstellen.")
        return
        
    final_df = pd.DataFrame(report_daten)
    
    
    try:
        os.makedirs(output_ordner, exist_ok=True)
        output_path = os.path.join(output_ordner, "finaler_report.xlsx")
        final_df.to_excel(output_path, index=False)
        print(f"\n--- Finaler Report erfolgreich erstellt! ---")
        print(f"Gespeichert unter: {output_path}")
    except Exception as e:
        print(f"Fehler beim Speichern des finalen Reports: {e}")
