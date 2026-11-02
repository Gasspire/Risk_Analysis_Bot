import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from config import CHATS_DIR


class ChatRepository:
    def __init__(self, chats_dir: str = CHATS_DIR):
        self.chats_dir = chats_dir
        os.makedirs(self.chats_dir, exist_ok=True)

    def _sanitize_filename(self, name: str) -> str:
        """Rimuove caratteri non validi per i filesystem Windows/Linux."""
        cleaned = re.sub(r'[\\/*?:"<>|]', "", name).strip()
        return cleaned if cleaned else f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _get_path(self, name: str) -> str:
        sanitized = self._sanitize_filename(name)
        if not sanitized.endswith(".json"):
            sanitized += ".json"
        return os.path.join(self.chats_dir, sanitized)

    def list_chats(self) -> List[str]:
        """
        Restituisce l'elenco dei nomi delle chat disponibili (senza estensione .json),
        ordinati per data di ultima modifica (dal più recente al più vecchio).
        """
        if not os.path.exists(self.chats_dir):
            return []

        files = [
            f for f in os.listdir(self.chats_dir)
            if f.endswith(".json") and os.path.isfile(os.path.join(self.chats_dir, f))
        ]

        # Ordina per tempo di ultima modifica decrescente
        files.sort(
            key=lambda f: os.path.getmtime(os.path.join(self.chats_dir, f)),
            reverse=True
        )

        return [os.path.splitext(f)[0] for f in files]

    def load_chat(self, name: str) -> List[Dict[str, str]]:
        """Carica la cronologia messaggi dal file corrispondente."""
        filepath = self._get_path(name)
        if not os.path.exists(filepath):
            return []

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict) and "messages" in data:
                    return data.get("messages", [])
        except Exception as e:
            print(f"Errore durante il caricamento della chat '{name}': {e}")

        return []

    def save_chat(self, name: str, messages: List[Dict[str, str]]) -> None:
        """Salva la cronologia messaggi nel file specificato."""
        filepath = self._get_path(name)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(messages, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Errore durante il salvataggio della chat '{name}': {e}")

    def create_chat(self, custom_name: Optional[str] = None) -> str:
        """
        Crea un nuovo file di chat vuoto. Se il nome non è specificato,
        usa una convenzione temporale (es. Chat_2026-11-02_11-50-00).
        Restituisce il nome assegnato (senza estensione).
        """
        if custom_name and custom_name.strip():
            base_name = self._sanitize_filename(custom_name.strip())
        else:
            base_name = f"Chat_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        final_name = base_name
        counter = 1
        while os.path.exists(self._get_path(final_name)):
            final_name = f"{base_name}_{counter}"
            counter += 1

        self.save_chat(final_name, [])
        return final_name

    def delete_chat(self, name: str) -> bool:
        """Elimina il file della chat specificata."""
        filepath = self._get_path(name)
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                return True
            except Exception as e:
                print(f"Errore durante l'eliminazione della chat '{name}': {e}")
        return False

    def rename_chat(self, old_name: str, new_name: str) -> Tuple[bool, str]:
        """
        Rinomina una chat esistente.
        Restituisce una tupla (successo: bool, nome_finale_o_errore: str).
        """
        clean_new = self._sanitize_filename(new_name).strip()
        if not clean_new:
            return False, "Il nuovo nome non è valido o contiene solo caratteri non ammessi."

        old_path = self._get_path(old_name)
        new_path = self._get_path(clean_new)

        if not os.path.exists(old_path):
            return False, f"La chat '{old_name}' non esiste."

        if os.path.abspath(old_path) == os.path.abspath(new_path):
            return True, clean_new

        if os.path.exists(new_path):
            return False, f"Esiste già una chat con il nome '{clean_new}'."

        try:
            os.rename(old_path, new_path)
            return True, clean_new
        except Exception as e:
            return False, f"Errore durante la rinomina del file: {e}"


