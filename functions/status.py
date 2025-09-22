import os
import json
from config import input_ordner 

STATUS_FILE_NAME = "_status.json"
STATUS_FILE_DIRECTORY = input_ordner

# Erstellt die Status-JSON-Datei im  Verzeichnis, falls sie noch nicht existiert.
def status_setup():
    status_file_path = os.path.join(STATUS_FILE_DIRECTORY, STATUS_FILE_NAME)
    if not os.path.exists(status_file_path):
        try:
            initial_status = {} 
            with open(status_file_path, 'w', encoding='utf-8') as f:
                json.dump(initial_status, f, ensure_ascii=False, indent=4)
            print(f"Statusdatei '{status_file_path}' wurde erfolgreich erstellt.")
        except Exception as e:
            print(f"Fehler beim Erstellen der Statusdatei '{status_file_path}': {e}")
    else:
        print(f"Statusdatei '{status_file_path}' existiert bereits.")

# Prüft, ob ein spezifischer Report für einen bestimmten 'stage_key' bereits verarbeitet wurde.
def load_status(report_filename, stage_key):
    status_file_path = os.path.join(STATUS_FILE_DIRECTORY, STATUS_FILE_NAME)
    status_data = {}
    if os.path.exists(status_file_path):
        try:
            with open(status_file_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warnung: Statusdatei '{status_file_path}' ist korrupt. Nehme leeren Status an.")
        except Exception as e:
            print(f"Fehler beim Laden der Statusdatei '{status_file_path}': {e}. Nehme leeren Status an.")
    
    processed_files_for_stage = status_data.get(stage_key, [])
    if not isinstance(processed_files_for_stage, list):
        # Falls der stage_key existiert, aber keine Liste ist (Datenfehler). Kam mal vor, aber dann nie wieeder. sicher ist sicher.
        print(f"Warnung: Daten für stage_key '{stage_key}' in Statusdatei sind keine Liste. Behandle als leer.")
        processed_files_for_stage = []
        
    return report_filename in processed_files_for_stage

# Speichert, dass ein spezifischer Report für einen bestimmten 'stage_key' verarbeitet wurde.
def save_status(report_filename, stage_key):
    status_file_path = os.path.join(STATUS_FILE_DIRECTORY, STATUS_FILE_NAME)
    status_data = {}
    # Lädt zuerst den aktuellen Gesamtstatus, um andere Stages nicht zu überschreiben
    if os.path.exists(status_file_path):
        try:
            with open(status_file_path, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Warnung: Statusdatei '{status_file_path}' beim Speichern korrupt gefunden. Erstelle neu für diesen Eintrag.")
            status_data = {} # Bei korrupter Datei neu anfangen (oder andere Fehlerbehandlung). Passiert.
        except Exception as e:
            print(f"Fehler beim Laden der Statusdatei für Speicherung '{status_file_path}': {e}. Erstelle neu für diesen Eintrag.")
            status_data = {}

    # Stellt sicher, dass der Stage-Key existiert und eine Liste ist
    processed_list_for_stage = status_data.get(stage_key, [])
    if not isinstance(processed_list_for_stage, list):
        # print(f"Warnung: Stage-Key '{stage_key}' war keine Liste. Initialisiere neu.")
        processed_list_for_stage = []

    if report_filename not in processed_list_for_stage:
        processed_list_for_stage.append(report_filename)
        processed_list_for_stage.sort() 
        status_data[stage_key] = processed_list_for_stage
        
        # Speichert das gesamte (aktualisierte) Status-Dictionary
        try:
            with open(status_file_path, 'w', encoding='utf-8') as f:
                json.dump(status_data, f, ensure_ascii=False, indent=4)
        except IOError as e:
            print(f"Fehler beim Schreiben der Statusdatei '{status_file_path}': {e}")
        except Exception as e:
            print(f"Ein unerwarteter Fehler beim Speichern der Statusdatei '{status_file_path}': {e}")
