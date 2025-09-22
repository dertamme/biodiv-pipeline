import os
from dotenv import load_dotenv
from functions.analyze_measures import analyze_measures_and_smartness
from functions.AI_clustering import fuehre_top_down_klassifizierung_durch
from functions.LDA_report import enrich_report_with_metadata
from functions.actions_clusterind_LDA import fuehre_lda_analyse_durch
from functions.clustering import  fuehre_vollstaendige_analyse_durch
from functions.clustering_planb import fuehre_zero_shot_klassifizierung_durch
from functions.deduplicate_statements import deduplicate_globally_per_file
from functions.final_orchestra import erstelle_finalen_report
from functions.find_actions_and_metrics import extract_details_from_passages
from functions.lda_cluster_names import name_lda_clusters
from functions.plots import erstelle_unternehmens_analyse
from functions.remove_empty_passages import bereinige_leere_passagen
from functions.robust_matching import behebe_zuordnungsfehler
from functions.screenshots import generate_screenshots
from functions.statistics import generate_company_jsons
from functions.status import status_setup
from functions.summary import summarize_actions_and_metrics
from functions.summary_stats import generate_global_summary
from functions.text_validation_gemini import text_validation_gemini
from functions.text_extraction import text_extraction
from functions.check_pdfs import clean_report_folder
from config import input_ordner, text_passages_ordner, relevant_text_passages_ordner, analyse_ordner
#from openai import OpenAI

    
ocr_verwenden = False


def main():
    if not os.path.isdir(input_ordner):
        os.makedirs(input_ordner, exist_ok=True)
        
    if not os.path.isdir(text_passages_ordner):
        os.makedirs(text_passages_ordner, exist_ok=True)
        
    if not os.path.isdir(relevant_text_passages_ordner):
        os.makedirs(relevant_text_passages_ordner, exist_ok=True)
          
    if not os.path.isdir(analyse_ordner):
        os.makedirs(analyse_ordner, exist_ok=True)
            
 
# > Erstelle die Status-Datei, um doppelte Bearbeitungen bei späterem Ausführen zu vermeiden.
    status_setup()
    
# > Entfernt Firmen, welche nicht im STOXX 600 Katalog (2025) sind
    clean_report_folder(input_ordner, "matching/sample_summary.xlsx")

    
# ========= Actions Identifikation  =========
# > Beginne mit Identifikation relevanter Stellen (+/- 5 Sätze) anhand von Keywords
#     print(">>>> Starte mit text_extraction <<<<< ")
#     text_extraction (input_ordner, text_passages_ordner, ocr_verwenden)
# # > Prüfe, ob innerhalb der Stellen, wo die Keywords stehen, auch Maßnahmen oder Metriken bzgl BioDiv genannt werden, oder ob nur das Keyword genannt wird. Wenn ja, gib die Action/Metric +/- 2 Sätze zurück (5 Sätze insg.).
#     print(">>>> Starte mit text_validation_gemini <<<<< ")
#     text_validation_gemini(text_passages_ordner,relevant_text_passages_ordner)
# # > Entfernt alle nicht mehr relevanten Textpassagen.
#     print(">>>> Starte mit bereinige_leere_passagen <<<<< ")
#     bereinige_leere_passagen(relevant_text_passages_ordner)
# # > Sucht nach Actions / Metrics innerhalb jeder Passage. Rückgabe nur ein Satz.
#     print(">>>> Starte mit extract_details_from_passages <<<<< ")
#     extract_details_from_passages(relevant_text_passages_ordner)
# # > Entfernt doppelte Einträge
#     print(">>>> Starte mit deduplicate_globally_per_file <<<<< ")
#     deduplicate_globally_per_file(relevant_text_passages_ordner)
# # > Fasse die relevanten Aktionen / Metriken zusammen
#     summarize_actions_and_metrics(relevant_text_passages_ordner, )
# # ======= Ende Actions Identifikation  =======


   
    
# # ========= Clustering LDA  =========
#     fuehre_lda_analyse_durch(input_ordner=relevant_text_passages_ordner, speicherordner="text_passages/analyse/LDA/fulltext")
#     name_lda_clusters ("text_passages/analyse/LDA/fulltext/LDA/lda_themen_report.xlsx", "text_passages/analyse/LDA/fulltext/LDA/")
#     enrich_report_with_metadata("text_passages/analyse/LDA/fulltext/LDA/clusters_final.xlsx","matching/sample_summary.xlsx", "text_passages/analyse/LDA/fulltext/LDA/" )
#     erstelle_unternehmens_analyse(angereicherter_report_pfad="text_passages/analyse/LDA/finaler_report_angereichert.xlsx", output_ordner="text_passages/analyse/LDA")
# # ======= ENDE LDA Clustering ========


# # ========= Clustering mit AI =========
#     print(">>>> Starte mit fuehre_top_down_klassifizierung_durch <<<<< ")
#     fuehre_top_down_klassifizierung_durch(relevant_text_passages_ordner, "matching/sample_summary.xlsx", "text_passages/analyse/AI")
#     final_report_path = "text_passages/analyse/AI/Top_Down_Analyse/top_down_klassifizierungs_report.xlsx"
#     behebe_zuordnungsfehler(report_path=final_report_path, summary_path="matching/sample_summary.xlsx")
# # ========= ENDE AI Clustering ========


# # ========= Hugging Face =========
#     fuehre_zero_shot_klassifizierung_durch(relevant_text_passages_ordner, "matching/sample_summary.xlsx", "text_passages/analyse/ZS")
# # ====== Ende Hugging Face =======


# # ========= UMAP Clustering =========
# # > Erstellt Kategorien und Überkategorien für die Actions
#     fuehre_vollstaendige_analyse_durch(input_ordner=relevant_text_passages_ordner,speicherordner="text_passages/analyse/UMAP/")
# # > Fasst die Ergebnisse pro Unternehmen zusammen 
#     erstelle_finalen_report(json_input_ordner=relevant_text_passages_ordner, cluster_ergebnis_xlsx_pfad="text_passages/analyse/UMAP/finaler_report.xlsx", output_ordner="text_passages/analyse/UMAP/" )
# # > Fasst die Ergebnisse pro Unternehmen zusammen (nicht LDA)
#     erstelle_finalen_report(json_input_ordner=relevant_text_passages_ordner, cluster_ergebnis_xlsx_pfad="text_passages/analyse/UMAP/finaler_report.xlsx", output_ordner="text_passages/analyse/UMAP/" )
# # ======  ENDE UMAP Clustering ======

   


# # ========= VISUALS & Statistics =========
#     final_report_path = "text_passages/analyse/AI/Top_Down_Analyse/top_down_klassifizierungs_report.xlsx"
#     global_summary_output_path = "text_passages/analyse/AI/globaler_summary_report.xlsx"
#     generate_global_summary(data_path=final_report_path,output_path=global_summary_output_path)


#     json_output_folder = "text_passages/analyse/AI/JSON_Reports"
#     print(">>>> Starte mit generate_company_jsons <<<<< ")
#     generate_company_jsons(data_path=final_report_path,output_folder=json_output_folder)

#     screenshots_output_folder = "text_passages/analyse/AI/Screenshots"
#     pdf_reports_folder = "input"
#     print(">>>> Starte mit generate_screenshots <<<<< ")
#     generate_screenshots(report_path=final_report_path,pdf_folder=pdf_reports_folder,output_folder=screenshots_output_folder)
# # ====== Ende VISUALS =======




# # ========= Berechne Anteile neue / alte Aussagen & SMART Ziele =========
#     daten_ordner = "matching/aussagen"
#     ergebnisse_ordner = daten_ordner
#     analyze_measures_and_smartness(daten_ordner, ergebnisse_ordner)
# # ====== Ende Berechne Anteile neue / alte Aussagen & SMART Ziele =======






if __name__ == "__main__":
    main()