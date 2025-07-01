import os
import json
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from typing import Dict, Any, Set, Tuple, List, Optional
from config import RESULTS_DIR
from utils import assert_file_exists

EX = Namespace("http://example.org/vocab#")
BASE = "http://example.org/persona/"

INPUT_DIR = RESULTS_DIR
OUTPUT_DIR = f"{RESULTS_DIR}/rdf_processed"

MAIN_RDF_FILE = os.path.join(OUTPUT_DIR, "personas-db.ttl")
UNPROCESSED_JSON_FILE = os.path.join(OUTPUT_DIR, "unprocessed_personas-db.json")

VERBOSE = False
PROCESS_ALL_PERSONAS = True
DELETE_UNPROCESSED_JSONS = True

FIELD_DEFINITIONS = {
    "ability to speak english": ("string", "abilityToSpeakEnglish"),
    "age": ("integer", "age"),
    "ancestry": ("string", "ancestry"),
    "big five scores": ("special_nested", "bigFiveScores"),
    "citizenship": ("string", "citizenship"),
    "class of worker": ("string", "classOfWorker"),
    "cognitive difficulty": ("string", "cognitiveDifficulty"),
    "defining quirks": ("string", "definingQuirks"),
    "detailed job description": ("string", "detailedJobDescription"),
    "disability": ("string", "disability"),
    "education": ("string", "education"),
    "employment status": ("string", "employmentStatus"),
    "family presence and age": ("string", "familyPresenceAndAge"),
    "fertility": ("string", "fertility"),
    "health insurance": ("string", "healthInsurance"),
    "hearing difficulty": ("string", "hearingDifficulty"),
    "household language": ("string_or_list", "householdLanguage"),
    "household type": ("string", "householdType"),
    "ideology": ("string", "ideology"),
    "income": ("float", "income"),
    "industry category": ("string", "industryCategory"),
    "lifestyle": ("string", "lifestyle"),
    "mannerisms": ("string", "mannerisms"),
    "marital status": ("string", "maritalStatus"),
    "occupation category": ("string", "occupationCategory"),
    "personal time": ("string", "personalTime"),
    "place of birth": ("string", "placeOfBirth"),
    "political views": ("string", "politicalViews"),
    "race": ("string", "race"),
    "religion": ("string", "religion"),
    "sex": ("string", "sex"),
    "veteran status": ("string", "veteranStatus"),
    "vision difficulty": ("string", "visionDifficulty"),
    "description": ("string", "description"),
}

def load_json_file(filepath: str) -> Optional[Dict[str, Any]]:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                if VERBOSE:
                    print(f"ERROR: JSON file '{filepath}' did not contain a dictionary at its root. Found: {type(data)}")
                return None
            return data
    except FileNotFoundError:
        if VERBOSE:
            print(f"ERROR: File not found at '{filepath}'.")
        return None
    except json.JSONDecodeError as e:
        if VERBOSE:
            print(f"ERROR: Invalid JSON format in '{filepath}': {e}")
        return None
    except Exception as e:
        if VERBOSE:
            print(f"ERROR: An unexpected error occurred while loading '{filepath}': {e}")
        return None

def save_json_file(filepath: str, data: Dict[str, Any]):
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"CRITICAL ERROR: Could not save JSON to '{filepath}': {e}")

def process_single_field_to_rdf(g: Graph, person_uri: URIRef, value: Any, intended_type: str, rdf_predicate_name: str, persona_id: int) -> bool:
    conversion_successful = True
    if value is None:
        return conversion_successful

    predicate = EX[rdf_predicate_name]

    if intended_type == "string":
        g.add((person_uri, predicate, Literal(str(value), datatype=XSD.string)))
    elif intended_type == "integer":
        try:
            g.add((person_uri, predicate, Literal(int(value), datatype=XSD.integer)))
        except (ValueError, TypeError):
            if VERBOSE:
                print(f"WARNING (Persona {persona_id}): Type conversion failed for predicate '{rdf_predicate_name}' with value '{value}' (expected integer). This persona will NOT be added to the main RDF graph.")
            g.add((person_uri, predicate, Literal(str(value), datatype=XSD.string)))
            conversion_successful = False
    elif intended_type == "float":
        try:
            g.add((person_uri, predicate, Literal(float(value), datatype=XSD.float)))
        except (ValueError, TypeError):
            if VERBOSE:
                print(f"WARNING (Persona {persona_id}): Type conversion failed for predicate '{rdf_predicate_name}' with value '{value}' (expected float). This persona will NOT be added to the main RDF graph.")
            g.add((person_uri, predicate, Literal(str(value), datatype=XSD.string)))
            conversion_successful = False
    elif intended_type == "string_or_list":
        if isinstance(value, list):
            for item in value:
                g.add((person_uri, predicate, Literal(str(item), datatype=XSD.string)))
        else:
            g.add((person_uri, predicate, Literal(str(value), datatype=XSD.string)))

    return conversion_successful

def warn_on_unrecognized_keys(person_id: int, data: Dict[str, Any]):
    unrecognized_keys = []
    for key in data.keys():
        if key not in FIELD_DEFINITIONS:
            unrecognized_keys.append(key)

    if unrecognized_keys:
        if VERBOSE:
            print(f"\nWARNING (ID {person_id}): Persona data contains unrecognized keys that will be skipped:")
            print(f"  Unrecognized: {unrecognized_keys}")
            print(f"  Keys in JSON: {sorted(list(data.keys()))}")
            print(f"  Expected (from FIELD_DEFINITIONS): {sorted(list(FIELD_DEFINITIONS.keys()))}\n")

            for uk in unrecognized_keys:
                if uk.strip() != uk:
                    print(f"  HINT: Unrecognized key '{uk}' has leading/trailing whitespace. Consider stripping it.")
                elif uk.lower() in [k.lower() for k in FIELD_DEFINITIONS.keys()] and uk not in FIELD_DEFINITIONS:
                     print(f"  HINT: Key '{uk}' might have incorrect casing compared to FIELD_DEFINITIONS.")

def convert_persona_json_to_rdf_graph(person_id: int, data: Dict[str, Any]) -> Tuple[Optional[Graph], bool]:
    if not isinstance(data, dict) or not data:
        if VERBOSE:
            print(f"ERROR (ID {person_id}): Persona data is empty or not a dictionary. Cannot process.")
        return None, False

    warn_on_unrecognized_keys(person_id, data)

    person_uri = URIRef(f"{BASE}{person_id}")
    source_uri = URIRef(f"https://huggingface.co/datasets/proj-persona/PersonaHub/viewer/persona/train?row={person_id}")

    g = Graph()
    g.bind("ex", EX)
    g.bind("xsd", XSD)
    g.add((person_uri, RDF.type, EX.Persona))
    g.add((person_uri, EX.source, source_uri))

    conversion_warnings_occurred = False

    try:
        for key, value in data.items():
            if key == "big five scores":
                bfs_data = value if isinstance(value, dict) else {}
                for trait, score in bfs_data.items():
                    if score is not None:
                        cleaned_trait = trait.replace(" ", "")
                        trait_predicate_name = cleaned_trait[0].lower() + cleaned_trait[1:] if cleaned_trait else ""
                        trait_uri = EX[trait_predicate_name]
                        g.add((person_uri, trait_uri, Literal(str(score), datatype=XSD.string)))
            elif key in FIELD_DEFINITIONS:
                definition = FIELD_DEFINITIONS.get(key)
                if definition and definition[0] != "special_nested":
                    intended_type, rdf_predicate_name = definition
                    field_successful = process_single_field_to_rdf(g, person_uri, value, intended_type, rdf_predicate_name, person_id)
                    if not field_successful:
                        conversion_warnings_occurred = True

        if conversion_warnings_occurred:
            return None, True

        return g, False

    except Exception as e:
        if VERBOSE:
            print(f"ERROR (Persona {person_id}): RDF conversion failed. Details: {e}")
        return None, False

def convert_and_get_result_wrapper(person_id: int, filepath: str) -> Tuple[int, Optional[Graph], str]:
    assert_file_exists(filepath)
    
    data = load_json_file(filepath)
    if data is None:
        return person_id, None, "load_error"

    try:
        graph, conversion_warnings_occurred = convert_persona_json_to_rdf_graph(person_id, data)

        if graph is not None:
            return person_id, graph, "success"
        elif conversion_warnings_occurred:
            return person_id, None, "conversion_warning"
        else:
            return person_id, None, "conversion_error_or_empty_data" 
    except Exception as e:
        if VERBOSE:
            print(f"CRITICAL ERROR: Uncaught exception during parallel processing of persona {person_id} from {filepath}: {e}")
        return person_id, None, "unhandled_exception"

def load_processed_persona_ids_from_rdf(filepath: str) -> Set[int]:
    processed_ids = set()
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return processed_ids

    g_temp = Graph()
    try:
        g_temp.parse(filepath, format='turtle')
    except Exception as e:
        if VERBOSE:
            print(f"WARNING: Could not parse existing main RDF file '{filepath}'. It might be corrupted or malformed. Error: {e}")
        return processed_ids

    for s, _, _ in g_temp.triples((None, RDF.type, EX.Persona)):
        try:
            person_id_str = str(s).split('/')[-1]
            if person_id_str.isdigit():
                processed_ids.add(int(person_id_str))
            else:
                if VERBOSE:
                    print(f"WARNING (File: {filepath}): Malformed Persona URI segment encountered: '{person_id_str}' from URI '{s}'. Skipping.")
        except ValueError:
            if VERBOSE:
                print(f"WARNING (File: {filepath}): Malformed Persona URI encountered: {s}. Skipping.")
        except Exception as e:
            if VERBOSE:
                print(f"WARNING (File: {filepath}): Unexpected error processing URI '{s}': {e}. Skipping.")
    return processed_ids

def load_unprocessed_personas_data(filepath: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        return {}
    return load_json_file(filepath) or {}

def save_unprocessed_personas_data(filepath: str, data: Dict[str, Any]):
    save_json_file(filepath, data)

def append_graphs_to_main_rdf(main_graph: Graph, new_graphs: List[Graph], main_rdf_filepath: str):
    if not new_graphs:
        print("INFO: No new graphs to append to the main RDF file.")
        return

    for g_new in new_graphs:
        main_graph += g_new

    temp_path = main_rdf_filepath + ".tmp"
    try:
        main_graph.serialize(destination=temp_path, format='turtle')
        os.replace(temp_path, main_rdf_filepath)
        print(f"INFO: Successfully saved updated main RDF graph to '{main_rdf_filepath}'.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to save or replace main RDF file '{main_rdf_filepath}'. Data might be incomplete. Error: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_all_personas():
    print(f"\n--- Starting Persona Processing ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    main_rdf_graph = Graph()
    try:
        if os.path.exists(MAIN_RDF_FILE) and os.path.getsize(MAIN_RDF_FILE) > 0:
            main_rdf_graph.parse(MAIN_RDF_FILE, format='turtle')
            print(f"INFO: Loaded existing main RDF graph from '{MAIN_RDF_FILE}'. Contains {len(main_rdf_graph)} triples.")
        else:
            print(f"INFO: '{MAIN_RDF_FILE}' not found or is empty. Starting with an empty main RDF graph.")
    except Exception as e:
        if VERBOSE:
            print(f"WARNING: Could not parse existing '{MAIN_RDF_FILE}'. Starting with an empty graph. Error: {e}")
    main_rdf_graph.bind("ex", EX)
    main_rdf_graph.bind("xsd", XSD)
    
    processed_ids_in_main_rdf = load_processed_persona_ids_from_rdf(MAIN_RDF_FILE)
    unprocessed_personas_data = load_unprocessed_personas_data(UNPROCESSED_JSON_FILE)

    all_json_files_in_input_dir = [
        os.path.join(INPUT_DIR, f)
        for f in os.listdir(INPUT_DIR)
        if f.startswith("persona_") and f.endswith(".json")
    ]
    if not all_json_files_in_input_dir:
        print(f"ERROR: No 'persona_*.json' files found in '{INPUT_DIR}'. Please check the INPUT_DIR path.")
        return

    candidate_files_for_processing = []

    for filepath in all_json_files_in_input_dir:
        filename = os.path.basename(filepath)
        try:
            person_id_str = filename.split("_")[1].split(".")[0]
            person_id = int(person_id_str)

            if person_id not in processed_ids_in_main_rdf:
                candidate_files_for_processing.append((person_id, filepath))
        except (ValueError, IndexError):
            if VERBOSE:
                print(f"WARNING: Skipping malformed filename: '{filename}'. Cannot extract persona ID. Please ensure filenames are in 'persona_ID.json' format (e.g., 'persona_123.json').")

    files_to_process_now = sorted(list(set(candidate_files_for_processing)), key=lambda x: x[0])

    print(f"\n--- Processing Run Summary ---")
    print(f"Total JSON files found in '{INPUT_DIR}': {len(all_json_files_in_input_dir)}")
    print(f"Personas already successfully in '{MAIN_RDF_FILE}': {len(processed_ids_in_main_rdf)}")
    print(f"Personas listed as unprocessed in '{UNPROCESSED_JSON_FILE}' at start: {len(unprocessed_personas_data)}")
    print(f"Will attempt to process {len(files_to_process_now)} unique personas (new or previously unprocessed).")

    if not files_to_process_now:
        print("INFO: No new or previously unprocessed personas found to process in this run.")
        print("\n--- Processing Run Complete ---")
        return

    newly_successful_graphs = []

    num_workers = os.cpu_count() or 4
    print(f"INFO: Using {num_workers} parallel processes for persona conversion.")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(convert_and_get_result_wrapper, pid, fp): (pid, fp)
                   for pid, fp in files_to_process_now}

        for future in tqdm(futures, total=len(files_to_process_now), desc="Converting Personas to RDF"):
            person_id, original_filepath = futures[future]
            try:
                _, result_graph_or_none, status = future.result()

                if status == "success":
                    newly_successful_graphs.append(result_graph_or_none)
                    if str(person_id) in unprocessed_personas_data:
                        del unprocessed_personas_data[str(person_id)]
                else:
                    failed_persona_data = load_json_file(original_filepath)
                    if failed_persona_data is not None:
                        if str(person_id) not in unprocessed_personas_data or unprocessed_personas_data[str(person_id)].get("processing_status") != status:
                            failed_persona_data["processing_status"] = status
                            unprocessed_personas_data[str(person_id)] = failed_persona_data
                    else:
                        if VERBOSE:
                            print(f"WARNING: Could not reload original JSON for failed/skipped persona {person_id} at {original_filepath}. Marking with internal error in unprocessed_personas.json.")
                        unprocessed_personas_data[str(person_id)] = {"error_reloading_original_json": "File not found or corrupted during failure handling.", "original_filepath": original_filepath, "processing_status": status}

            except Exception as e:
                if VERBOSE:
                    print(f"CRITICAL ERROR: Uncaught exception in main process during result retrieval for persona {person_id} from {original_filepath}: {e}")
                unprocessed_personas_data[str(person_id)] = {"critical_unhandled_error": str(e), "original_filepath": original_filepath, "processing_status": "unhandled_exception_in_main_thread"}

    append_graphs_to_main_rdf(main_rdf_graph, newly_successful_graphs, MAIN_RDF_FILE)
    save_unprocessed_personas_data(UNPROCESSED_JSON_FILE, unprocessed_personas_data)

    print(f"\n--- Processing Run Complete ---")
    final_processed_count = load_processed_persona_ids_from_rdf(MAIN_RDF_FILE)
    final_unprocessed_count_data = load_unprocessed_personas_data(UNPROCESSED_JSON_FILE)
    print(f"Total personas now successfully in '{MAIN_RDF_FILE}': {len(final_processed_count)}")
    print(f"Total personas now listed as unprocessed in '{UNPROCESSED_JSON_FILE}': {len(final_unprocessed_count_data)}")

    if final_unprocessed_count_data:
        print(f"\nACTION REQUIRED: There are still {len(final_unprocessed_count_data)} unprocessed personas.")
        print(f"Please check '{UNPROCESSED_JSON_FILE}' for details on what failed.")
        if DELETE_UNPROCESSED_JSONS:
            print("The source JSON files for these unprocessed personas will be deleted automatically.")
        else:
            print("To clean up source JSON files for these unprocessed personas, set DELETE_UNPROCESSED_JSONS = True and run again.")
    else:
        print("\nAll remaining persona JSON files in the input directory have been successfully processed or were already processed!")

def delete_unprocessed_source_jsons():
    print(f"\n--- Starting Deletion of Unprocessed Source JSONs ---")

    unprocessed_data = load_unprocessed_personas_data(UNPROCESSED_JSON_FILE)

    if not unprocessed_data:
        print("INFO: No unprocessed personas found in the database. Nothing to delete.")
        return

    deleted_count = 0
    for persona_id_str in tqdm(unprocessed_data.keys(), desc="Deleting Unprocessed JSONs"):
        try:
            filepath_to_delete = os.path.join(INPUT_DIR, f"persona_{persona_id_str}.json")

            if os.path.exists(filepath_to_delete):
                os.remove(filepath_to_delete)
                deleted_count += 1
        except OSError as e:
            print(f"ERROR: Could not delete file '{filepath_to_delete}': {e}")
        except Exception as e:
            print(f"ERROR: An unexpected error occurred while trying to delete persona {persona_id_str}'s file: {e}")

    print(f"INFO: Deleted {deleted_count} source JSON files corresponding to unprocessed personas from '{INPUT_DIR}'.")
    print(f"--- Deletion Complete ---")

if __name__ == "__main__":
    if PROCESS_ALL_PERSONAS:
        process_all_personas()
        
    if DELETE_UNPROCESSED_JSONS:
        delete_unprocessed_source_jsons()