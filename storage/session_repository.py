import json
import os
from config import SESSION_FILE
from models import SessionStateData


class SessionRepository:
    def __init__(self, filepath: str = SESSION_FILE):
        self.filepath = filepath

    def save(self, state: SessionStateData) -> None:
        """Serializza l'intero oggetto di stato (inclusa la chat history e lo stato del bot) su disco."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(), f, indent=4, ensure_ascii=False)

    def load(self, default_context: str = "") -> SessionStateData:
        """Deserializza lo stato dal disco, se presente. Altrimenti restituisce uno stato predefinito."""
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return SessionStateData.model_validate(data)
            except Exception as e:
                print(f"Errore durante il caricamento della sessione salvata: {e}")

        return SessionStateData(general_context=default_context)