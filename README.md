#  Risk Assessment Copilot (Risk Analysis Bot)

**Risk Assessment Copilot** è una piattaforma interattiva per l'analisi e la gestione del rischio cyber e fisico, sviluppata con **Streamlit**, **LangChain**, **Ollama** e **ChromaDB**. 

Il sistema è allineato con la metodologia dello standard internazionale **ISO/IEC 27005** e con i principi di sicurezza, integrità e resilienza previsti dal **GDPR**. Combina un **motore RAG (Retrieval-Augmented Generation)** con un **agente conversazionale a stati finiti** per guidare gli analisti di sicurezza nella valutazione oggettiva delle minacce, nella classificazione dei controlli e nella prioritizzazione del rischio.

---

##  Scenario di Test e Validazione: Smart Hospital (ENISA)

Il sistema è stato testato su uno scenario ad alta criticità: la protezione delle infrastrutture sanitarie connesse, prendendo come riferimento lo studio ufficiale dell'Agenzia dell'Unione europea per la cybersecurity (ENISA):

>  **ENISA Publication:** [Cyber Security and Resilience for Smart Hospitals](https://www.enisa.europa.eu/publications/cyber-security-and-resilience-for-smart-hospitals)

### Dati e Contesto di Test
- **Asset sanitari e IoT medicale (IoMT):** Dispositivi medici connessi (infusion pumps, monitor multiparametrici), server PACS/RIS, sistemi EHR/cartella clinica elettronica, sottostazioni di rete e sistemi BMS (*Building Management Systems*).
- **Vettori di minaccia analizzati:** Attacchi ransomware con blocco dei servizi di emergenza, compromissione dell'integrità dei dati diagnostici, accessi non autorizzati da remoto, interruzioni della continuità operativa e guasti fisici/ambientali.
- **Misure di sicurezza indicizzate:** Segmentazione di rete (VLAN sanitarie), sistemi EDR/MDR, backup immutabili offline, autenticazione a più fattori (MFA), protocolli di disaster recovery e ridondanza elettrica.

---

##  Caratteristiche Principali

- **Assistente Conversazionale a Stati (Risk Agent Executor):**
  - Routing intelligente degli intenti tramite LLM (`rag_info`, `single_threat_analysis`, `batch_assessment`, `reevaluate_filtered`, `discuss_modify`, `asset_extract`).
  - Macchina a stati (`IDLE`, `AWAITING_THREAT_DETAILS`, `DISCUSSING_THREAT`) per raccogliere proattivamente dettagli mancanti (minaccia, asset, contromisure).
  - Possibilità di discutere, rifinire e calibrare le valutazioni direttamente via chat in linguaggio naturale.

- **Pipeline Ibrida di Estrazione Asset (NLP + Clustering Semantico + LLM):**
  - **Estrazione Linguistica (spaCy):** Identificazione automatica di sostantivi, nomi propri, acronimi e sintagmi nominali composti (`noun_chunks`) dai documenti architetturali di sistema (`.txt`, `.md`) e dal contesto generale, con lemmatizzazione e rimozione delle stopword.
  - **Embedding e Spherical K-Means Dinamico:** Vettorizzazione dei termini candidati normalizzata con norma L2 e raggruppamento tramite Spherical K-Means (basato su cosine similarity) con selezione adattiva del numero ottimale di cluster $K$ mediante **Silhouette Score**.
  - **Filtro di Coesione e Fusione (Merge):** Riassorbimento controllato dei cluster singleton affini, scarto dei cluster con bassa coesione semantica interna e fusione iterativa dei cluster con centroidi altamente simili.
  - **Selezione, Deduplicazione e Descrizione via LLM:** Valutazione dei cluster tramite LLM per isolare esclusivamente gli asset meritevoli di protezione, deduplicazione lessicale e semantica dei nomi estratti e generazione automatica di una descrizione contestuale per ciascun asset sulla base dei documenti caricati.

- **Metodologia di Risk Assessment Rigorosa:**
  - **Classificazione Controlli di Sicurezza:** Suddivisione automatica delle misure fornite in **Preventive** (riducono la probabilità prima dell'evento) e **Mitigative** (riducono l'impatto o agevolano la risposta/ripristino).
  - **Calcolo Likelihood (1 - 5):** Valutazione basata su frequenza storica, facilità di esecuzione e controlli preventivi.
  - **Calcolo Impact (1 - 5):** Valutazione dello scenario peggiore (*worst-case*), impatto sulla vita umana, continuità dei servizi essenziali, riservatezza dei dati e controlli mitigativi.
  - **Risk Factor (1 - 25):** Calcolo della matrice di rischio $P \times I$ con indicazione visiva del livello (Very Low, Low, Medium, High, Critical).

- **Valutazione Multi-Run con Moda Statistica:**
  - Supporto per esecuzioni multiple per minaccia (1–10 iterazioni).
  - Calcolo della **moda statistica** per neutralizzare la varianza delle risposte del modello.
  - Catena LLM di **sintesi coerente** per produrre motivazioni allineate al punteggio finale consolidato.

- **Motore RAG con ChromaDB:**
  - Caricamento automatico di documentazione da `./knowledge_base` o `./documents_rag` e upload dinamico di file di policy/inventari asset (`.txt`) direttamente dall'interfaccia.
  - Indicizzazione vettoriale con `nomic-embed-text` e ricerca di similarità semantica per contestualizzare ogni analisi.

- **Valutazioni Batch e Filtrate:**
  - Importazione di inventari minacce da file JSON.
  - Esecuzione massiva dell'assessment su tutto il dataset con un solo click.
  - Rivalutazione mirata e filtrata per specifici asset o condizioni di rischio (es. impatto > 3).

- **Personalizzazione, Override Manuale e Persistenza:**
  - Inserimento di regole personalizzate per Likelihood e Impact da parte dell'analista.
  - Override manuale dei punteggi nel pannello di reportistica per garantire pieno controllo all'operatore umano (*Human-in-the-Loop*).
  - Persistenza automatica della sessione (stato, chat history, minacce) su `session_backup.json`.

---

## Architettura del Progetto

```
Risk_Analysis_Bot/
│
├── app.py                      # Interfaccia utente Streamlit (Chat, Configurazione & Report)
├── config.py                   # Parametri globali (URL Ollama, modelli, percorsi, sessione)
├── models.py                   # Modelli Pydantic, Enums di stato/intento e schemi di output
├── prompt.py                   # Template dei prompt di sistema, few-shot examples e schemi
├── requirements.txt            # Dipendenze Python del progetto
├── session_backup.json         # Salvataggio persistente dello stato globale della sessione
│
├── core/                       # Motore logico di AI, RAG, Asset Extraction e Risk Assessment
│   ├── asset_extractor.py      # Pipeline di estrazione asset (spaCy, Spherical K-Means, LLM)
│   ├── llm_chains.py           # Factory per la creazione delle catene LangChain strutturate
│   ├── rag_engine.py           # Gestione ChromaDB, text-splitters ed embeddings
│   └── risk_analyzer.py        # Logica di calcolo del rischio, iterazioni e sintesi
│
├── llm_handler/                # Orchestrazione del dialogo e gestione comandi
│   ├── executor.py             # RiskAgentExecutor: routing degli intenti e macchina a stati
│   ├── handlers.py             # Handler specifici per i vari intenti conversazionali
│   └── formatting.py           # Funzioni di utilità per formattazione e indicatori visivi
│
├── storage/                    # Persistenza e gestione dello storico
│   ├── chat_repository.py      # Gestione salvataggio, esportazione e caricamento sessioni di chat
│   └── session_repository.py   # Serializzazione e deserializzazione JSON dello stato dell'applicazione
│
├── chats/                      # Cronologie conversazionali salvate ed esportabili (.json)
├── system_documents/           # Documenti di sistema e architettura per l'estrazione asset (.txt, .md)
├── documents_rag/              # Knowledge Base: tassonomie NIST, scenari e policy Smart Hospital (.txt)
└── chroma_db/                  # Database vettoriale locale persistente generato da ChromaDB
```

---

## Prerequisiti

1. **Python 3.10+** (consigliato Python 3.10 o 3.11).
2. **Ollama**:
   - Assicurarsi che l'istanza di Ollama sia raggiungibile (in locale `http://localhost:11434` o remoto).
   - Scaricare i modelli necessari:
     ```bash
     ollama pull llama3.1
     ollama pull nomic-embed-text
     ```

---

## Installazione e Configurazione

### 1. Clona il repository e posizionati nella cartella di progetto:
```bash
cd Risk_Analysis_Bot
```

### 2. Crea ed attiva un ambiente virtuale:
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

### 3. Installa le dipendenze e il modello linguistico spaCy:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configura i parametri (`config.py`):
Modifica il file `config.py` in base al tuo ambiente:
```python
BASE_URL = "http://localhost:11434"      # Indirizzo del server Ollama
MODEL_NAME = "llama3.1"                 # Modello LLM per analisi, estrazione asset e chat
EMBEDDING_MODEL = "nomic-embed-text"    # Modello di embedding per ChromaDB e clustering asset
TEMPERATURE = 0.1                       # Bassa per garantire determinismo e riproducibilità
CORE_KB_DIR = "./knowledge_base"         # Directory della knowledge base iniziale
ASSET_DOCS_DIR = "./system_documents"   # Directory dei documenti di sistema per l'estrazione asset
CHROMA_PERSIST_DIR = "./chroma_db"      # Cartella di persistenza di Chroma
SESSION_FILE = "session_backup.json"    # File di backup della sessione
```

---

## Avvio dell'Applicazione

Avvia l'applicazione Streamlit:
```bash
streamlit run app.py
```

L'interfaccia web si aprirà automaticamente nel browser all'indirizzo `http://localhost:8501`.

---

## Guida all'Utilizzo

### 1. Configurazione Iniziale (Sidebar)
- **Contesto Generale:** Inserisci informazioni relative all'organizzazione (es. settore ospedaliero, infrastruttura cloud, sede geografica).
- **Importa Minacce (JSON):** Carica un elenco di minacce strutturato (vedi sezione *Formato File JSON*).
- **Aggiorna Contesto RAG:** Carica documenti di testo (`.txt`) contenenti policy aziendali, piani di emergenza o specifiche tecniche per indicizzarli nel Vector Store.
- **Documenti di Sistema per Estrazione Asset:** Carica file `.txt` o `.md` che descrivono l'architettura del sistema; i file vengono salvati in `system_documents/` per alimentare la pipeline di individuazione degli asset.
- **Regole di Valutazione:** Definisci eventuali vincoli o policy personalizzate per il calcolo di Likelihood e Impact.
- **Numero Iterazioni per Minaccia:** Imposta quante volte il modello deve valutare ciascuna minaccia (default: `1`). Impostando un valore $>1$ il sistema calcola la **moda statistica** e genera una sintesi motivazionale coerente.

### 2. Assistente Analisi (Chat)
Puoi interagire in linguaggio naturale con l'assistente per:
- **Analisi di una singola minaccia:**
  > *"Analyze the threat: Ransomware attack on the electronic health record (EHR) server with daily offline backups and perimeter next-gen firewall."*  
  Se mancano campi necessari, l'agente chiederà esplicitamente le informazioni mancanti.
- **Estrazione Automatica degli Asset:**
  > *"Extract all protectable assets from the system documentation."*  
  Avvia la pipeline di estrazione e mostra il riepilogo delle metriche intermedie (termini candidati, $K$ ottimale, Silhouette Score, cluster selezionati) insieme alla lista degli asset estratti e descritti.
- **Domande sulla Knowledge Base (RAG):**
  > *"What are the internal recovery procedures and resilience requirements in case of a power blackout in the ICU?"*
- **Esecuzione Batch:**
  > *"Evaluate all uploaded threats."*
- **Rivalutazione Mirata:**
  > *"Re-evaluate only threats with an impact score greater than or equal to 4."*  
  > *"Re-evaluate all threats targeting network infrastructure and telemetry assets."*
- **Discussione e Modifica di una Minaccia Esistente:**
  > *"Let's discuss the threat 'Fire in server room'. I'd like to lower the likelihood to 1 because we installed certified optical-thermal smoke detectors."*

### 3. Report Minacce
- Visualizza l'elenco ordinato delle minacce per **Risk Factor** decrescente.
- Consulta per ciascuna voce:
  - Misure preventive e mitigative identificate e separate.
  - Motivazioni analitiche di Likelihood e Impact.
  - Punteggio totale con badge visivo.
- **Override Manuale:** Modifica puntualmente i valori di Likelihood e Impact e premi **"Salva e applica modifiche manuali"**.

### 4. Asset Estratti
- Avvia direttamente dall'interfaccia la pipeline di estrazione automatica tramite il pulsante **"Estrai Asset dai Documenti di Sistema"**.
- Consulta la lista globale degli asset censiti completa delle relative descrizioni contestuali generate a partire dalla documentazione di sistema.
- Aggiungi manualmente nuovi asset con descrizione personalizzata, rimuovi singole voci o svuota l'intera lista globale.

---

## Formato File JSON Minacce

Il caricamento delle minacce da file richiede un array JSON contenente esclusivamente i campi fondamentali: `threat`, `affected_assets` e `security_measures`:

```json
[
  {
    "threat": "Ransomware infection via phishing",
    "affected_assets": "EHR Server, PACS Database, Medical Workstations",
    "security_measures": "Next-Gen Firewall, Daily Immutable Backups, EDR Agent, MFA on VPN"
  },
  {
    "threat": "Unauthorized physical access to server room",
    "affected_assets": "Core Switches, Telemetry Gateway, Patient Monitoring Unit",
    "security_measures": "Biometric Access Control, CCTV Monitoring, Intrusion Detection Alarm"
  },
  {
    "threat": "HVAC failure and thermal overload",
    "affected_assets": "Datacenter Racks, On-premise Storage Arrays",
    "security_measures": "Dual Redundant Cooling Units, Environmental Heat Sensors, Automatic Alerting"
  }
]
```

---

## Tecnologie Utilizzate

- **[Streamlit](https://streamlit.io/)**: Interfaccia utente web reattiva, modulare e intuitiva.
- **[LangChain](https://www.langchain.com/)**: Catene prompt/LLM e orchestrazione con output strutturato Pydantic.
- **[Ollama](https://ollama.com/)**: Runtime locale/privato per l'inferenza di modelli open weights (LLaMA 3.1) a garanzia di sovranità e privacy dei dati (GDPR-compliant).
- **[ChromaDB](https://www.trychroma.com/)**: Vector store open-source ad alte prestazioni per la memorizzazione e il recupero semantico della Knowledge Base.
- **[spaCy](https://spacy.io/)**: Libreria NLP per l'analisi morfosintattica, lemmatizzazione ed estrazione dei sintagmi nominali candidati dai documenti di sistema.
- **[scikit-learn](https://scikit-learn.org/) & [NumPy](https://numpy.org/)**: Normalizzazione L2, inizializzazione K-Means++, calcolo della cosine similarity e valutazione del Silhouette Score per il clustering semantico degli asset.
- **[Pydantic](https://docs.pydantic.dev/)**: Validazione dei dati, conformità degli schemi e tipizzazione rigorosa.