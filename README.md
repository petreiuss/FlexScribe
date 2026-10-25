# FlexScribe

Applicazione web locale per psicoterapeuti: trascrive le registrazioni delle sedute con Whisper, genera sintesi cliniche con un modello linguistico locale (Gemma tramite llama.cpp) e mostra l'andamento dei processi psicologici (hexaflex ACT) seduta dopo seduta.

**Tutto gira sul computer: nessun audio, trascrizione o sintesi viene inviato in rete.** Il server ascolta solo su `127.0.0.1`.

## Funzioni

- **Pazienti e sedute** come normali cartelle su disco (`Pazienti/<Paziente>/<AAAA-MM-GG_titolo>/`). I file audio copiati direttamente nella cartella di un paziente diventano automaticamente una seduta datata.
- **Trascrizione** con Whisper large-v3-turbo su scheda grafica Intel (OpenVINO), circa 8 minuti per un'ora di audio; fallback su CPU con whisper.cpp.
- **Sintesi cliniche** con modelli ("skill") personalizzabili in `skills/`: sintesi di seduta (CBT di terza generazione), nota SOAP, lettera al medico curante, punti chiave.
- **Modifica con Gemma** (riscrive solo le sezioni interessate), **modifica a mano**, **controllo grammaticale** (che non tocca le citazioni del paziente) e **cronologia delle versioni** con annullamento.
- **Scheda dei processi** per seduta (accettazione, defusione, momento presente, sé come contesto, valori, azione impegnata, autocompassione), con punteggi proposti da Gemma da confermare o correggere.
- **Andamento** per paziente: grafico hexaflex, linee di tendenza, tabella delle sedute.

## Requisiti

- Windows 10/11 (l'app usa alcune funzioni specifiche di Windows)
- Python da 3.12 a 3.14 (testato con 3.13). Python 3.15 non è ancora supportato: OpenVINO non è ancora disponibile per quella versione
- Consigliata una scheda grafica Intel integrata recente (Core Ultra / Iris Xe) con driver aggiornati
- [llama.cpp](https://github.com/ggml-org/llama.cpp) con il comando `llama` (build Vulkan) e un modello Gemma in formato GGUF, ad esempio `google/gemma-4-E4B-it-qat-q4_0-gguf`
- Circa 8 GB di spazio libero (modelli) e 16 GB di RAM

## Installazione

```bat
git clone <url-del-repository>
cd flexscribe
avvia.bat
```

Al primo avvio `avvia.bat`:
1. crea l'ambiente Python (`.venv`) e installa `requirements.txt`;
2. scarica il modello Whisper per la scheda grafica (`OpenVINO/whisper-large-v3-turbo-int8-ov`, circa 800 MB) in `models/`;
3. avvia il server e apre il browser su <http://127.0.0.1:8765>.

Al primo avvio viene creato anche `config.json` con le impostazioni predefinite: controlla soprattutto `patients_dir`, `llm_exe` e `llm_model`.

Per avere l'icona **FlexScribe** sul Desktop, esegui una volta `crea-collegamento.bat`: il collegamento avvia l'app con la finestra del server ridotta a icona (chiuderla spegne l'app).

La prima trascrizione dopo ogni avvio richiede circa 30 secondi in più per caricare il modello sulla scheda grafica.

### Gemma (sintesi)

L'app avvia da sola `llama serve` quando serve e lo chiude dopo `llm_idle_minutes` minuti di inattività. Scarica il modello GGUF (con `llama` o da Hugging Face) e indica in `config.json`:
- `llm_exe`: percorso di `llama.exe` (trovato automaticamente se è nel PATH)
- `llm_model`: percorso del file `.gguf` (trovato automaticamente se è nella cache di Hugging Face)

Per verificare quale scheda grafica vede llama.cpp: `llama serve --list-devices`.

### Fallback CPU (facoltativo)

Per trascrivere senza scheda grafica, metti una build di [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (`whisper-cli.exe` e un modello `ggml-*.bin`) nella cartella `whisper.cpp/` oppure indica la cartella in `whisper_dir`. È circa 8 volte più lento.

## Configurazione (`config.json`)

| Chiave | Significato |
|---|---|
| `patients_dir` | Cartella dei pazienti (predefinita: `Documenti\Pazienti`) |
| `default_model` | Modello di trascrizione predefinito (`ov:<modello>:GPU` oppure un file `ggml-*.bin`) |
| `language` | Lingua delle sedute (`it`) |
| `whisper_prompt` | Vocabolario suggerito a Whisper (termini ACT, CBT) |
| `whisper_dir` | Cartella di whisper.cpp per il fallback su CPU |
| `llm_exe`, `llm_model` | Percorsi di `llama.exe` e del modello Gemma |
| `llm_device`, `llm_gpu_layers` | Dispositivo per Gemma (`Vulkan0` = scheda grafica, `none` = CPU) |
| `llm_ctx` | Contesto massimo del modello in token (32768 basta per circa 2 ore di seduta) |
| `llm_idle_minutes` | Minuti di inattività dopo cui Gemma viene chiuso per liberare memoria |
| `llm_spec` | Decodifica speculativa (`ngram-simple`, oppure `""` per disattivarla) |
| `default_skill` | Modello di sintesi predefinito |
| `auto_process_card` | Crea la scheda dei processi dopo ogni nuova sintesi |
| `host`, `port` | Indirizzo del server (lasciare `127.0.0.1`) |

## Personalizzare le sintesi

Ogni file Markdown in `skills/` è un modello di sintesi: l'intestazione contiene `nome` e `descrizione`, il resto è il prompt. Per aggiungerne uno, copia un file esistente e modificalo: compare subito nel menu, senza riavviare.

`skills/_contesto.md` descrive l'orientamento del clinico (CBT di terza generazione: ACT, mindfulness, CFT, DBT, MCT…) ed è condiviso da tutti i modelli.

## Struttura dei dati

```
Pazienti/
  Rossi Mario/
    2026-10-18_Prima_seduta/
      audio.m4a                  registrazione originale
      transcript.txt             trascrizione (modificabile)
      transcript.json            segmenti con orari
      sintesi_<modello>.md       sintesi cliniche
      processi.json              scheda dei processi (proposti e confermati)
      versioni/                  versioni precedenti delle sintesi
      meta.json                  data, titolo, cronologia
```

I dati dei pazienti restano nella cartella `patients_dir`, fuori dal repository. Il file `.gitignore` esclude comunque i file tipici delle sedute.

## Avvertenze

Le sintesi e i punteggi sono **bozze generate da un modello linguistico** e possono contenere errori: vanno sempre verificati dal clinico. L'app non sostituisce il giudizio clinico. Proteggi i dati dei pazienti, ad esempio con la crittografia del disco (BitLocker) e con backup.
