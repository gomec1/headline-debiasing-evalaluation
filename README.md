# BiasScore Evaluation App

> **Bachelor Thesis** | Bachelor of Science in Digital Business & AI
> Bern University of Applied Sciences (BFH) — Business School
> Author: Carlos Gomez

Forschungsprototyp fuer die Bachelorarbeit zur Evaluation von LLMs bei der Erkennung und Reformulierung von Bias in News-Headlines. Die App fokussiert zwei Bias-Arten: **Linguistic Bias** und **Hyperpartisanship**. Fuer beide Bias-Arten gibt es jeweils eine Detection-Pipeline mit Bewerter-Prompt und eine Mitigation-Pipeline mit Reformulierer-Prompt. Mehrere LLMs koennen parallel verglichen und die Ergebnisse als CSV, Excel und Markdown exportiert werden.

## Installation

Voraussetzungen:

- Python >= 3.10
- Windows, macOS oder Linux
- Optional fuer lokale Modelle: [Ollama](https://ollama.com)

Setup:

```powershell
cd bias_eval_app
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Start:

```powershell
python eval_app.py
```

Die App startet standardmaessig unter `http://127.0.0.1:7860`.

### Sicherheit / Git-Hygiene

Diese Dateien duerfen committet werden:

- `eval_app.py`
- `README.md`
- `.gitignore`
- `requirements.txt`
- `config.example.json`
- `prompts/*.txt`

Diese Dateien duerfen nicht committet werden:

- `config.json`
- `secrets.json`
- `.env`
- `ergebnisse_*.csv`
- `ergebnisse_*.xlsx`
- `ergebnisse_*.md`
- Exportdateien mit sensiblen Headlines

Beim ersten Start nach dem Update erkennt die App alte `config.json`-Dateien mit Klartext-Keys automatisch. Die Keys werden nach `secrets.json` verschoben, und `config.json` wird bereinigt.

Neue Einrichtung:

```powershell
Copy-Item config.example.json config.json
python eval_app.py
```

Danach die API-Keys in Tab 1 eintragen. Die App speichert sie automatisch in `secrets.json`.

Falls ein API-Key versehentlich committet wurde: Key sofort beim Provider invalidieren, neuen Key erstellen und das Repository bereinigen, z. B. mit `git filter-branch` oder dem BFG Repo-Cleaner.

## Schnellstart

Empfohlener Ablauf:

1. Tab 1: LLMs konfigurieren.
2. Tab 2.1: eigene Headlines eingeben, manuell labeln und LLM-Bewertung starten.
3. Tab 2.2: Lyu-CSV hochladen, Spalten pruefen und Validierung starten.
4. Tab 3.1: Ergebnisse aus 2.1 uebernehmen und Reformulierung testen.
5. Tab 3.2: Ergebnisse aus 2.2 uebernehmen und Hyperpartisan-Reformulierung testen.
6. Tab 4: Dashboard aktualisieren und alle Ergebnisse exportieren.

## Tab 1 - LLM-Konfiguration

Tab 1 verwaltet alle Modelle, die in den Analysen verwendet werden.

Unterstuetzte Provider:

| Provider | Client | Basis-URL |
|---|---|---|
| OpenAI | OpenAI-Client | leer |
| Anthropic | Anthropic-Client | leer |
| Google (Gemini) | OpenAI-kompatibel | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Groq | OpenAI-kompatibel | `https://api.groq.com/openai/v1` |
| Together AI | Together SDK | API-Key aus `TOGETHER_API_KEY` |
| Ollama (lokal) | OpenAI-kompatibel | `http://localhost:11434/v1` |

### Groq

Groq-Keys werden unter https://console.groq.com erstellt. In der App:

- Provider: `Groq`
- Basis-URL: leer lassen
- Beispiel-Modelle: `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`, `gemma2-9b-it`, `openai/gpt-oss-20b`

Wenn die Basis-URL leer bleibt, setzt die App automatisch `https://api.groq.com/openai/v1`.

### Together AI

Together-Modelle koennen ohne Code-Aenderung in Tab 1 hinzugefuegt werden:

- Provider: `Together AI`
- Modell-ID: z. B. `meta-llama/Llama-3.3-70B-Instruct-Turbo` oder `mistralai/Mixtral-8x7B-Instruct-v0.1`
- API-Key: im UI eintragen oder alternativ als Umgebungsvariable `TOGETHER_API_KEY`

Beispiel:

```powershell
$env:TOGETHER_API_KEY="dein-key"
python eval_app.py
```

Together nutzt zuerst den in Tab 1 gespeicherten Key aus `secrets.json`. Wenn dort kein Key vorhanden ist, wird als Fallback `TOGETHER_API_KEY` gelesen. Together nutzt ausserdem einen In-Memory-Session-Cache pro laufender App-Session. Gleiche Kombinationen aus Modell, System-Prompt und User-Message werden nicht erneut an die API gesendet.

## Prompts-Ordner

Die Prompts liegen in `prompts/`:

- `prompts/bewerter_linguistic_bias.txt`
- `prompts/bewerter_hyperpartisan.txt`
- `prompts/reformulierer_linguistic_bias.txt`
- `prompts/reformulierer_hyperpartisan.txt`

Die Dateien enthalten reinen Prompt-Text ohne Python-Variablenzuweisung. Sie koennen waehrend des Betriebs editiert werden. Danach in Tab 1 den Button **Prompts neu laden** klicken. Der naechste LLM-Call nutzt dann die aktualisierte Datei.

## Prompt Caching

Prompt Caching bedeutet: Ein Anbieter merkt sich einen gleichbleibenden Prompt fuer kurze Zeit. In dieser App betrifft das vor allem die System-Prompts fuer Bewerter und Reformulierer. Wenn derselbe Prompt bei vielen Headlines wiederverwendet wird, muss der Anbieter ihn nicht jedes Mal voll neu verarbeiten. Das kann Kosten senken und manchmal auch die Antwortzeit verbessern.

| Provider | Caching-Art | Umsetzung in der App | Typische Ersparnis |
|---|---|---|---|
| Anthropic Claude | explizit | `cache_control: {"type": "ephemeral"}` auf dem System-Prompt | Cache-Lesen ca. 10% des normalen Input-Preises |
| OpenAI | automatisch | keine API-Aenderung, Cache-Status wird geloggt | cached Tokens ca. 50% guenstiger |
| Google Gemini | automatisch fuer Gemini 2.5+ | keine API-Aenderung, Cache-Status wird geloggt | bis zu ca. 75% guenstiger |
| Groq | automatisch fuer einzelne Modelle | keine API-Aenderung, Hinweis bei nicht unterstuetzten Modellen | cached Tokens ca. 50% guenstiger |
| Ollama | lokal | kein Provider-Caching noetig | keine API-Kosten |

Debug-Meldungen im Terminal zeigen Cache-Hits, zum Beispiel:

```text
[DEBUG] Anthropic Cache-HIT: 4500 Tokens aus Cache gelesen (gespart!)
[DEBUG] OpenAI Cache-HIT: 1200 Tokens aus Cache gelesen (gespart!)
```

Bei Anthropic werden die aktuellen Prompts korrekt fuer Caching markiert. Die API nutzt den Cache aber erst, wenn die Mindestgroesse erreicht ist. Die jetzigen Prompts sind relativ kurz; deshalb kann es sein, dass zunaechst keine Cache-Hits erscheinen. Bei Groq unterstuetzen nur bestimmte Modelle Prompt Caching, aktuell unter anderem `moonshotai/kimi-k2-instruct`, `openai/gpt-oss-20b` und `openai/gpt-oss-120b`. Fuer andere Groq-Modelle gibt die App im Terminal einen Hinweis aus.

## JSON-Ausgabe und Validierung

Prompt-only JSON ist keine Garantie fuer valides oder schema-konformes JSON. Die Bewerter-Pipeline nutzt deshalb mehrere Schutzschichten: native Structured Outputs mit `response_schema`, wenn ein Provider sie zuverlaessig unterstuetzt, sonst JSON-Object-Mode, robuste lokale JSON-Extraktion, zentrale Schema-Validierung und begrenzte Retry-/Repair-Aufrufe.

Gemini kann ueber das offizielle Google GenAI SDK mit `response_mime_type="application/json"` und `response_schema` angesprochen werden, sofern `google-genai` installiert ist. Der vorhandene OpenAI-kompatible Gemini-Endpunkt bleibt als Fallback erhalten; dieser Modus ist JSON-Mode, aber kein echtes Gemini `response_schema`.

Bei OpenAI wird nach Moeglichkeit JSON-Schema-Mode genutzt. OpenAI-kompatible Endpunkte wie Gemini-Fallback, Groq, Together AI, Llama-Endpunkte und Ollama unterscheiden sich je nach Anbieter und Modell. Wenn ein Endpoint kein Schema oder keinen JSON-Object-Mode akzeptiert, bleibt die lokale Validierung mit Retry/Repair verpflichtend.

Die LLMs liefern bei Bewerter-Outputs nur noch primaere Dimensionsbewertungen: einzelne Dimensionsscores, `dimension_evidence` und eine kurze finale `reasoning`. Abgeleitete Felder wie `total_score`, `category` und bei Hyperpartisanship `binary_label` werden deterministisch lokal berechnet. Dadurch koennen logische Inkonsistenzen in Modelloutputs nicht mehr in die finalen Ergebnisse wandern.

Die lokale Bewerter-Validierung prueft alle Pflichtfelder, Score-Integer von 0 bis 3, `dimension_evidence` und `reasoning`. Falls ein Modell trotzdem `total_score`, `category` oder `binary_label` mitschickt, werden diese Werte ignoriert und lokal neu berechnet. JSON-Syntaxfehler koennen weiterhin auftreten und werden durch Extraktion, Validierung und Retry/Repair behandelt. Die finale Speicherung und alle Exporte enthalten weiterhin die bisherigen Felder `total_score`, `category` und `binary_label`, soweit sie fuer die Analyse relevant sind.

Reformulierer erhalten als Input immer das normalisierte Bewerter-Ergebnis inklusive lokal berechnetem Score, Kategorie und ggf. `binary_label`. Sie erzeugen selbst keine Scores, Kategorien, `dimension_evidence` oder `reasoning`, sondern nur `neutralized_headline`, `changed_terms`, `meaning_preservation`, `neutralization_summary` und `changed_meaning_risk`. Bewerter- und Reformulierer-Schemas sind bewusst getrennt, damit sich Bewertungs- und Rewrite-JSON nicht vermischen.

Jeder Analyse-Lauf erzeugt zusaetzlich zum normalen Textlog ein strukturiertes JSONL-Auditlog `logs/*_json_pipeline.jsonl`. Jede Zeile ist ein eigenes JSON-Objekt und enthaelt eine Audit-Schema-Version. Dort werden Raw-Output-Vorschauen vor Korrekturen, JSON-Extraktion, entfernte Markdown-Fences oder `<think>`-Bloecke, Newline-Reparaturen, Parse-Fehler, Schema-Validierung, Normalisierungen, lokal berechnete Felder, entfernte Zusatzfelder, Retry-Versuche und finale gueltige oder ungueltige Outputs protokolliert. Das gilt fuer Bewerter- und Reformulierer-Outputs sowie fuer den normalisierten Input an den Reformulierer.

Vollstaendige Raw-Outputs werden standardmaessig nicht gespeichert. Im Normalbetrieb protokolliert die App nur eine gekuerzte und redigierte Vorschau, die Laenge des urspruenglichen Outputs und ob gekuerzt wurde. Vollstaendiges Raw-Logging sollte nur fuer gezielte Debug-Laeufe aktiviert werden:

```powershell
$env:LOG_RAW_LLM_OUTPUTS="true"
$env:LOG_RAW_LLM_OUTPUT_LIMIT="1000"
python eval_app.py
```

Auch bei gekuerzten Raw-Outputs bleiben Korrekturdetails strukturiert sichtbar, z. B. Score-String zu Integer, ueberschriebene `total_score`/`category`/`binary_label`-Modellwerte, entfernte Felder und Retry-Gruende. API-Keys, Bearer-Tokens, Authorization-Header, Secrets und Passwortfelder werden vor normalen Logs und JSONL-Auditlogs maskiert. Auditlogs dienen der technischen Nachvollziehbarkeit und beeinflussen die wissenschaftlichen Metriken nicht; vollstaendige Raw-Outputs werden nicht in CSV- oder Excel-Ergebnisexporte geschrieben.

Lokaler Test ohne API-Calls:

```powershell
python -c "import eval_app as e; e.run_json_pipeline_selftest()"
python -c "import eval_app as e; e.run_json_pipeline_logging_selftest()"
python -c "import eval_app as e; e.run_json_pipeline_audit_security_selftest()"
python -c "import eval_app as e; e.run_state_and_export_selftest()"
```

## Tab 2 - Bewerter

### Analyse 2.1: Ground Truth - eigener annotierter Datensatz

Forschungsfrage: Wie gut stimmen LLMs mit der eigenen menschlichen Annotation fuer Linguistic Bias ueberein?

Ablauf:

1. Headlines eingeben.
2. Annotationstabelle erzeugen.
3. Jede Headline mit `Low`, `Medium` oder `High` labeln.
4. Mehrere LLMs auswaehlen.
5. Bewertung starten.

Outputs:

- Vergleichstabelle: Headline, eigenes Label, LLM-Labels, Scores, Uebereinstimmung
- Cohen's Kappa zwischen allen LLM-Paaren
- Precision, Recall, F1 pro LLM
- 2x2 Confusion Matrix pro LLM
- Bis zu 5 Fehlklassifikationen pro LLM

### Analyse 2.2: Externe Validierung - Lyu et al. (2024)

Forschungsfrage: Wie gut erkennen LLMs Hyperpartisanship gegen eine externe Ground Truth?

Datensatz: Lyu et al. (2024), `manually_labeled_data.csv`. Die App erkennt die Spalten `title` und `label` automatisch, falls vorhanden. Die Stichprobe wird mit `random_state=42` gezogen.

Mapping:

- `binary_label = non-hyperpartisan` -> 0
- `binary_label = hyperpartisan` -> 1
- Fallback: `Low` -> 0, `Medium/High` -> 1

Outputs:

- Vergleichstabelle
- Precision, Recall, F1 pro LLM
- 2x2 Confusion Matrix pro LLM
- Fehlklassifikationen

## Tab 3 - Reformulierer

Die Reformulierer-Analysen folgen einer Bewerter-Reformulierer-Bewerter-Pipeline. Zuerst liegt eine Bias-Analyse vor. Diese Analyse wird als `bias_analysis` an den Reformulierer uebergeben. Danach wird die reformulierte Headline erneut mit demselben Bewerter-Prompt bewertet.

### Analyse 3.1: Linguistic Bias

Modi:

- Modus A: Ergebnisse aus Analyse 2.1 uebernehmen. Es findet kein neuer Vorher-Bewerter-Call statt.
- Modus B: Neue Headlines eingeben und direkt bewerten.

Der Schwellenwert-Slider entscheidet, welche Headline-LLM-Paare reformuliert werden. Bei Score unterhalb der Schwelle bleibt die Zeile mit Status `unter Schwelle - nicht reformuliert` in der Tabelle.

Outputs:

- LLM
- Original
- Score vorher
- Kategorie vorher
- Reformuliert
- Score nachher
- Kategorie nachher
- Delta Score
- Kategorie-Reduktion
- Cosine Similarity

### Analyse 3.2: Hyperpartisanship

Analyse 3.2 ist analog zu 3.1, nutzt aber die Hyperpartisan-Prompts. Modus A uebernimmt Ergebnisse aus Analyse 2.2 inklusive Stichprobe. Modus B laedt eine neue CSV und bewertet sie direkt. Optional kann auf Headlines mit Lyu-Label `1` gefiltert werden.

## Tab 4 - Dashboard & Export

Tab 4 aggregiert die zuletzt berechneten Ergebnisse aus Tab 2 und Tab 3.

Der Button **Alle Ergebnisse exportieren** erzeugt:

- einzelne CSV-Dateien pro vorhandener Analyse
- eine kombinierte Excel-Datei mit mehreren Sheets
- eine Markdown-Datei mit methodischer Berechnungserklaerung

Dateinamen enthalten einen Zeitstempel im Format `YYYY-MM-DD_HHMM`.
Excel-Exports enthalten ein Sheet `Ergebnisse` mit den Ergebniszeilen und technischen JSON-Metadaten wie `json_status`, `json_warnings`, `correction_applied`, `retry_count` und `raw_output_available`, sofern diese vorhanden sind. Diese Felder dienen nur der Nachvollziehbarkeit und werden nicht fuer Kappa, Metriken oder Confusion-Matrizen verwendet. Vollstaendige Raw-Outputs und Raw-Previews werden nicht in Excel exportiert.

## Erzeugte Dateien

| Datei | Inhalt | Versionieren? |
|---|---|---|
| `config.example.json` | Beispielkonfiguration ohne Keys | ja |
| `config.json` | lokale Modell-Metadaten ohne Keys | nein |
| `secrets.json` | lokale API-Keys | nein |
| `prompts/*.txt` | Bewerter- und Reformulierer-Prompts | ja |
| `exports/ergebnisse_*.csv` | Analyseergebnisse | nein |
| `exports/ergebnisse_*.xlsx` | Excel-Exports | nein |
| `exports/ergebnisse_*.md` | Methodische Export-Erklaerung | nein |
| `logs/*.log` | Laufzeit-Logs pro gestarteter Analyse | nein |

## Analyse-Logs und Abbruch

Jeder Klick auf eine Analyse startet eine eigene Logdatei im Ordner `logs/`. Dort werden Start, Statusmeldungen, LLM-Debugmeldungen, Warnungen, Fehler und der Abschlussstatus gespeichert. Parallel entsteht ein strukturiertes JSONL-Auditlog fuer die JSON-Pipeline. Die Analyse-Tabs haben zusaetzlich einen Button **Analyse abbrechen**. Ein Abbruch stoppt die Gradio-Queue; ein bereits laufender externer API-Call kann technisch noch kurz bis zum Timeout oder zur Antwort weiterlaufen.

## Fehlerbehebung

### API-Key fehlt

In Tab 1 wird bei Modellen ohne Key `FEHLT` angezeigt. Den Key im API-Key-Feld nachtragen und das Modell erneut speichern.

### Groq funktioniert nicht

Provider `Groq` waehlen, Modell-ID exakt eintragen und Basis-URL leer lassen. Die App setzt die URL automatisch.

### Ollama: Verbindung verweigert

Pruefen, ob `ollama serve` laeuft und ob das Modell mit `ollama pull <modell>` installiert wurde.

### Kein JSON gefunden

Einzelne LLMs halten sich manchmal nicht an das geforderte JSON-Format. Die App protokolliert den Fehler fuer dieses Modell und setzt die restliche Analyse fort.

### Prompt-Datei fehlt

Die UI zeigt den fehlenden Dateinamen an. Datei im Ordner `prompts/` erstellen oder aus den vorhandenen Prompt-Dateien wiederherstellen.

### Cosine Similarity dauert beim ersten Mal

Beim ersten Aufruf wird `sentence-transformers/all-mpnet-base-v2` geladen. Danach bleibt das Modell im Speicher.

## Referenzen

- Lyu et al. (2024): Computational Assessment of Hyperpartisanship in News Titles.
- Menzner & Leidner (2024): BiasScanner.
- Raza et al. (2024): LLM Agreement und Evaluationsmetriken.
- Landis & Koch (1977): Interpretation von Cohen's Kappa.
- Recasens et al. (2013): Linguistic Bias und Framing.
- Hamborg et al. (2019): Automated Identification of Media Bias.

## Limitations

Die eigene Annotation ist klein und stammt von einem einzelnen menschlichen Annotator. Hyperpartisanship und Linguistic Bias sind verwandte, aber nicht identische Konzepte. Die Ergebnisse sind deshalb fuer eine Bachelorarbeit auswertbar, aber nicht als allgemeingueltige Benchmark zu verstehen.

## Smoke-Test-Walkthrough

1. `python eval_app.py` starten.
2. In Tab 1 ein Testmodell speichern und pruefen, dass kein Key in der Tabelle sichtbar ist.
3. In Tab 2.1 zwei bis drei Headlines labeln und mit einem oder zwei LLMs bewerten.
4. In Tab 2.2 eine kleine Lyu-kompatible CSV laden und eine Stichprobe bewerten.
5. In Tab 3.1 Modus A waehlen, Schwelle veraendern und Vorschau pruefen.
6. In Tab 3.1 Reformulierung starten und im Status pruefen, dass nur Reformulierer + Re-Bewertung laufen.
7. In Tab 3.2 Modus A oder B ausfuehren.
8. In Tab 4 Dashboard aktualisieren und Gesamtexport erzeugen.

## License

This project is licensed under the **MIT License** — see [LICENSE](https://github.com/gomec1/headline-debiasing-dataset/blob/main/LICENSE) for details.

## Academic Context

This repository accompanies the bachelor thesis:

> **KI-gestützte Entbiasierung von Headlines — Design, Implementation und Evaluation**
> Carlos Gomez
> Bachelor of Science in Digital Business & AI
> Bern University of Applied Sciences (BFH), Business School
> Supervised by: Prof. Ulrich Matter (IADSF, Departement W)
