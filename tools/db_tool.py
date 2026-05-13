import json
import os
from langchain_core.tools import tool

DB_FILE = os.path.join(os.path.dirname(__file__), "patient_db.json")

def _init_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump({
                "P12345": {
                    "name": "Jane Smith",
                    "history": ["Diagnosed with Type 2 Diabetes in 2021", "Allergic to Peanuts"]
                }
            }, f)

_init_db()

@tool
def get_patient_history(patient_id: str) -> str:
    """Retrieves the patient history from the EHR database using patient_id."""
    with open(DB_FILE, "r") as f:
        data = json.load(f)
    if patient_id in data:
        record = data[patient_id]
        return f"Patient Name: {record['name']}. History: {', '.join(record['history'])}"
    return f"Patient ID {patient_id} not found in the database."

@tool
def update_patient_record(patient_id: str, new_information: str) -> str:
    """Updates the patient record in the EHR database with new information."""
    with open(DB_FILE, "r") as f:
        data = json.load(f)
    
    if patient_id in data:
        data[patient_id]["history"].append(new_information)
    else:
        data[patient_id] = {"name": "Unknown", "history": [new_information]}
        
    with open(DB_FILE, "w") as f:
        json.dump(data, f)
        
    return f"Successfully updated record for patient {patient_id}."
