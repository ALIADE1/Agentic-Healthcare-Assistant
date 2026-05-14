import json
import os
from langchain_core.tools import tool

DB_FILE = os.path.join(os.path.dirname(__file__), "patient_db.json")

def _init_db():
    if not os.path.exists(DB_FILE) or os.path.getsize(DB_FILE) == 0:
        initial_data = {
            "P12345": {
                "age": 45,
                "gender": "Female",
                "blood_type": "A+",
                "chronic_conditions": ["Type 2 Diabetes", "Hypertension"],
                "allergies": ["Peanuts", "Penicillin"],
                "past_surgeries": ["Appendectomy (2010)"],
                "current_medications": ["Metformin", "Lisinopril"]
            },
            "P67890": {
                "age": 62,
                "gender": "Male",
                "blood_type": "O-",
                "chronic_conditions": ["Coronary Artery Disease"],
                "allergies": [],
                "past_surgeries": ["CABG (2018)", "Cholecystectomy (2005)"],
                "current_medications": ["Atorvastatin", "Aspirin", "Metoprolol"]
            },
            "P11223": {
                "age": 28,
                "gender": "Female",
                "blood_type": "B+",
                "chronic_conditions": ["Asthma"],
                "allergies": ["Dust Mites", "Latex"],
                "past_surgeries": [],
                "current_medications": ["Albuterol Inhaler", "Fluticasone"]
            },
            "P44556": {
                "age": 50,
                "gender": "Male",
                "blood_type": "AB+",
                "chronic_conditions": ["Chronic Kidney Disease"],
                "allergies": ["Sulfa Drugs"],
                "past_surgeries": ["Kidney Biopsy (2020)"],
                "current_medications": ["Losartan", "Erythropoietin"]
            },
            "P99887": {
                "age": 35,
                "gender": "Female",
                "blood_type": "O+",
                "chronic_conditions": ["Hypothyroidism"],
                "allergies": [],
                "past_surgeries": ["C-Section (2019)"],
                "current_medications": ["Levothyroxine"]
            },
            "P33445": {
                "age": 71,
                "gender": "Male",
                "blood_type": "A-",
                "chronic_conditions": ["Osteoarthritis", "COPD"],
                "allergies": ["Ibuprofen"],
                "past_surgeries": ["Right Knee Replacement (2022)"],
                "current_medications": ["Tiotropium", "Acetaminophen"]
            }
        }
        with open(DB_FILE, "w") as f:
            json.dump(initial_data, f, indent=4)

_init_db()

@tool
def get_patient_history(patient_id: str) -> str:
    """Retrieves the patient history from the EHR database using patient_id."""
    with open(DB_FILE, "r") as f:
        data = json.load(f)
    if patient_id in data:
        return json.dumps(data[patient_id], indent=2)
    return f"Patient ID {patient_id} not found in the database."

@tool
def update_patient_record(patient_id: str, new_information: dict) -> str:
    """Updates the patient record in the EHR database with new information."""
    with open(DB_FILE, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = {}
    
    if patient_id not in data or data.get(patient_id) is None:
        data[patient_id] = {
            "age": None,
            "gender": "",
            "blood_type": "",
            "chronic_conditions": [],
            "allergies": [],
            "past_surgeries": [],
            "current_medications": []
        }
        
    data[patient_id].update(new_information)
        
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)
        
    return f"Successfully updated record for patient {patient_id}."
