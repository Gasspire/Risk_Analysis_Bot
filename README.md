# 🛡️ Risk Assessment Copilot (Risk Analysis Bot)

**Risk Assessment Copilot** è una piattaforma interattiva per l'analisi e la gestione del rischio cyber e fisico, sviluppata con **Streamlit**, **LangChain**, **Ollama** e **ChromaDB**. 

Il sistema combina un **motore RAG (Retrieval-Augmented Generation)** con un **agente conversazionale a stati finiti** per guidare gli analisti di sicurezza nella valutazione qualitativa e quantitativa delle minacce, nella classificazione dei controlli e nel calcolo oggettivo del fattore di rischio.

---

## 🚀 Caratteristiche Principali

- **🤖 Assistente Conversazionale a Stati (Risk Agent Executor):**
  - Routing intelligente degli intenti tramite LLM (`rag_info`, `single_threat_analysis`, `batch_assessment`, `reevaluate_filtered`, `discuss_modify`).
  - Macchina a stati (`IDLE`, `AWAITING_THREAT_DETAILS`, `DISCUSSING_THREAT`) per raccogliere proattivamente dettagli mancanti (minaccia, asset, contromisure).
  - Possibilità di discutere, rifinire e modificare le valutazioni direttamente via chat in linguaggio naturale.

- **📊 Metodologia di Risk Assessment Rigorosa:**
  - **Classificazione Controlli di Sicurezza:** Suddivisione automatica delle misure fornite in **Preventive** (riducono la probabilità prima dell'evento) e **Mitigative** (riducono l'impatto o agevolano la risposta/ripristino).
  - **Calcolo Likelihood (1 - 5):** Valutazione basata su frequenza storica, facilità di esecuzione e controlli preventivi.
  - **Calcolo Impact (1 - 5):** Valutazione dello scenario peggiore (worst-case), impatto sulla vita umana/continuità aziendale e controlli mitigativi.
  - **Risk Factor (1 - 25):** Calcolo della matrice di rischio $P \times I$ con indicazione visiva del livello (Basso, Medio, Alto, Critico).

- **🔄 Valutazione Multi-Run con Moda Statistica:**
  - Supporto per esecuzioni multiple per minaccia (1–10 iterazioni).
  - Calcolo della **moda statistica** per neutralizzare la varianza delle risposte del modello.
  - Catena LLM di **sintesi coerente** per produrre motivazioni allineate al punteggio finale consolidato.

- **📚 Motore RAG con ChromaDB:**
  - Caricamento automatico di documentazione da `./knowledge_base` e upload dinamico di file di policy/inventari asset (`.txt`) direttamente dall'interfaccia.
  - Indicizzazione vettoriale con `nomic-embed-text` e ricerca di similarità per contestualizzare ogni analisi.

- **⚡ Valutazioni Batch e Filtrate:**
  - Importazione di inventari minacce da file JSON.
  - Esecuzione massiva dell'assessment su tutto il dataset con un solo click.
  - Rivalutazione mirata e filtrata per specifici asset o condizioni di rischio (es. impatto > 3).

- **🎛️ Personalizzazione, Override Manuale e Persistenza:**
  - Inserimento di regole personalizzate per Likelihood e Impact da parte dell'analista.
  - Override manuale dei punteggi nel pannello di reportistica.
  - Persistenza automatica della sessione (stato, chat history, minacce) su `session_backup.json`.

---

## 🏗️ Architettura del Progetto

```
Risk_Analysis_Bot/
│
├── app.py                      # Interfaccia utente principale Streamlit (Chat & Report)
├── config.py                   # Parametri globali (URL Ollama, modelli, directory, sessione)
├── models.py                   # Modelli Pydantic, Enums di stato/intento e schemi di output
├── prompt.py                   # Template dei prompt di sistema, few-shot examples e schemi
├── requirements.txt            # Dipendenze Python del progetto
├── session_backup.json         # Salvataggio persistente dello stato della sessione
│
├── core/                       # Motore logico di AI e RAG
│   ├── llm_chains.py           # Factory per la creazione delle catene LangChain strutturate
│   ├── rag_engine.py           # Gestione ChromaDB, text-splitters ed embeddings
│   └── risk_analyzer.py        # Logica di calcolo del rischio, iterazioni e sintesi
│
├── llm_handler/                # Orchestrazione del dialogo e gestione comandi
│   ├── executor.py             # RiskAgentExecutor: routing degli intenti e gestione stati
│   ├── handlers.py             # Handler specifici per i vari intenti conversazionali
│   └── formatting.py           # Funzioni di utilità per formattazione e indicatori visivi
│
├── storage/                    # Persistenza dati
│   └── session_repository.py   # Serializzazione e deserializzazione JSON della sessione
│
├── knowledge_base/             # (Opzionale) Documenti di contesto e policy iniziali (.txt)
└── chroma_db/                  # Database vettoriale persistito generato da Chroma
```

---

## 📋 Prerequisiti

1. **Python 3.10+** (consigliato Python 3.10 o 3.11).
2. **Ollama**:
   - Assicurarsi che l'istanza di Ollama sia raggiungibile (in locale `http://localhost:11434` o remoto).
   - Scaricare i modelli necessari:
     ```bash
     ollama pull llama3.1
     ollama pull nomic-embed-text
     ```

---

## ⚙️ Installazione e Configurazione

### 1. Clona il repository o posizionati nella cartella di progetto:
```bash
cd Risk_Analysis_Bot
```

### 2. Crea ed attiva un ambiente virtuale (consigliato):
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 3. Installa le dipendenze:
```bash
pip install -r requirements.txt
```

### 4. Configura i parametri (`config.py`):
Modifica il file `config.py` in base al tuo ambiente:
```python
BASE_URL = "http://localhost:11434"      # Indirizzo del server Ollama
MODEL_NAME = "llama3.1"                 # Modello LLM per analisi e chat
EMBEDDING_MODEL = "nomic-embed-text"    # Modello di embedding per ChromaDB
TEMPERATURE = 0.1                       # Temperatura (bassa per maggiore determinismo)
CORE_KB_DIR = "./knowledge_base"         # Directory della knowledge base iniziale
CHROMA_PERSIST_DIR = "./chroma_db"      # Cartella di persistenza di Chroma
SESSION_FILE = "session_backup.json"    # File di backup della sessione
```

---

## 🖥️ Avvio dell'Applicazione

Avvia l'applicazione Streamlit tramite terminale:
```bash
streamlit run app.py
```

L'interfaccia web si aprirà automaticamente nel browser predefinito all'indirizzo `http://localhost:8501`.

---

## 📖 Guida all'Utilizzo

### 1. Configurazione Iniziale (Sidebar)
- **Contesto Generale:** Inserisci informazioni relative all'organizzazione (es. settore ospedaliero, infrastruttura cloud, sede geografica).
- **Importa Minacce (JSON):** Carica un elenco di minacce strutturato (vedi sezione *Formato File JSON*).
- **Aggiorna Contesto RAG:** Carica documenti di testo (`.txt`) contenenti policy aziendali, piani di emergenza o specifiche tecniche per indicizzarli nel Vector Store.
- **Regole di Valutazione:** Definisci eventuali vincoli o policy personalizzate per il calcolo di Likelihood e Impact.
- **Numero Iterazioni per Minaccia:** Imposta quante volte il modello deve valutare ciascuna minaccia (default: `1`). Impostando un valore $>1$ il sistema calcola la **moda statistica** e genera una sintesi motivazionale.

### 2. Assistente Analisi (Chat)
Puoi interagire in linguaggio naturale con l'assistente per:
- **Analisi di una singola minaccia:**
  > *"Analizza la minaccia: Attacco ransomware sui server dei referti con backup offline e firewall perimetrale."*
  Se mancano campi necessari, il bot ti chiederà esplicitamente le informazioni mancanti.
- **Domande sulla Knowledge Base (RAG):**
  > *"Quali sono le procedure interne per la gestione dei blackout elettrici?"*
- **Esecuzione Batch:**
  > *"Esegui la valutazione di tutte le minacce caricate."*
- **Rivalutazione Mirata:**
  > *"Rivaluta solo le minacce con impatto maggiore o uguale a 4."*
  > *"Rivaluta le minacce relative agli asset di rete."*
- **Discussione e Modifica di una Minaccia Esistente:**
  > *"Discutiamo la minaccia 'Incendio'. Vorrei abbassare la likelihood a 1 perché abbiamo installato sensori termo-ottici."*

### 3. Report Minacce
- Visualizza l'elenco ordinato delle minacce per **Risk Factor** decrescente.
- Consulta per ciascuna voce:
  - Misure preventive e mitigative identificate.
  - Motivazioni analitiche di Likelihood e Impact.
  - Punteggio totale con badge visivo (🟢 Basso, 🟡 Medio, 🟠 Alto, 🔴 Critico).
- **Override Manuale:** Modifica puntualmente i valori di Likelihood e Impact e premi **"Salva e applica modifiche manuali"**.

---

## 📄 Formato File JSON Minacce

Il caricamento delle minacce da file supporta un array JSON con i seguenti campi:

```json
[
  {
    "id": "optional-uuid-string",
    "threat": "Ransomware infection",
    "affected_assets": "Database Server, Patient Records",
    "security_measures": "Next-Gen Firewall, Daily Offline Backups, Antivirus",
    "likelihood": 1,
    "impact": 1
  },
  {
    "threat": "Allagamento locale server",
    "affected_assets": "Rack principale, gruppo di continuità",
    "security_measures": "Sensori di allagamento a pavimento, pompe di drenaggio",
    "likelihood": 1,
    "impact": 1
  }
]
```

> **Nota:** I campi `likelihood`, `impact`, `preventive_measures` e `mitigative_measures` verranno calcolati e aggiornati automaticamente dal modello durante la valutazione.

---

## 🧮 Matrice del Rischio

Il fattore di rischio complessivo è dato dal prodotto:
$$\text{Risk Factor} = \text{Likelihood (1--5)} \times \text{Impact (1--5)}$$

| Valore Risk Factor | Livello | Indicatore |
| :---: | :---: | :---: |
| **$1 - 4$** | Rischio Basso | 🟢 Low Risk |
| **$5 - 9$** | Rischio Medio | 🟡 Medium Risk |
| **$10 - 15$** | Rischio Alto | 🟠 High Risk |
| **$16 - 25$** | Rischio Critico | 🔴 Critical Risk |

---

## 🛠️ Tecnologie Utilizzate

- **[Streamlit](https://streamlit.io/)**: Interfaccia utente web reattiva e modulare.
- **[LangChain](https://www.langchain.com/)**: Catene prompt/LLM con output strutturato Pydantic.
- **[Ollama](https://ollama.com/)**: Runtime locale/remoto per l'esecuzione di LLaMA 3.1 e modelli di embedding.
- **[ChromaDB](https://www.trychroma.com/)**: Vector store per la memorizzazione e il recupero della Knowledge Base documentale.
- **[Pydantic](https://docs.pydantic.dev/)**: Validazione e tipizzazione dello stato e dei payload strutturati.

---

## 📄 Licenza

Distribuito sotto licenza libera per scopi di sviluppo, ricerca e analisi di sicurezza. Consultare gli autori per utilizzi commerciali o specifici accordi organizzativi.

---

ATTENZIONE: Il README è stato creato completamente tramite AI e verifica che sia corretto e completo per l'uso che ne faremo, il bot è quasi completo ma ho bisogno di apportare alcune modifiche, il bot sarà utilizzato per l'analisi del rischio cyber e fisico, per l'esattezza analizzeremo minacce, asset e contromisure, poi calcoleremo la probabilità e l'impatto in base al rischio e infine calcoleremo il risk factor.