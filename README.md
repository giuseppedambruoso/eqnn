# EQNN — Equivariant Quantum Neural Network

[![CI](https://github.com/giuseppedambruoso/eqnn/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/giuseppedambruoso/eqnn/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](pyproject.toml)

Progetto di ricerca su reti neurali quantistiche (QNN) equivarianti, addestrate su MNIST tramite circuiti parametrici (PennyLane) e tracciate su [Weights & Biases](https://wandb.ai).

Ogni sezione qui sotto è spiegata in due modi:
- 🔰 **Se non hai esperienza di programmazione** — segui questi passi alla lettera, copiando i comandi così come sono.
- 👩‍💻 **Se sei uno sviluppatore** — versione sintetica con i dettagli tecnici.

---

## 1. Installazione

### 🔰 Guida per chi parte da zero

1. Installa **Docker Desktop** (include tutto il necessario): vai su https://www.docker.com/products/docker-desktop/, scaricalo e installalo, poi aprilo (deve rimanere in esecuzione in background — vedrai un'icona nella barra delle applicazioni).
2. Apri un terminale (su Windows: "WSL"/"Ubuntu" dal menu Start; su Mac: l'app "Terminale").
3. Scarica il progetto sul tuo computer:
   ```bash
   git clone git@github.com:giuseppedambruoso/eqnn.git
   cd eqnn
   ```
4. Crea il file delle impostazioni segrete a partire dal modello fornito:
   ```bash
   cp .env.example .env
   ```
5. Apri il file `.env` con un editor di testo semplice (es. `nano .env` nel terminale) e incolla la tua **API key di Weights & Biases**, che serve per vedere i grafici dei risultati in un sito web dedicato. La ottieni gratuitamente registrandoti su https://wandb.ai e copiandola da https://wandb.ai/authorize. Il file deve assomigliare a:
   ```
   WANDB_API_KEY=la-tua-chiave-qui
   WANDB_MODE=online
   WANDB_PROJECT=eqnn
   ```
   Salva e chiudi (`Ctrl+O`, invio, poi `Ctrl+X` in `nano`).
6. Costruisci l'ambiente software (la prima volta richiede qualche minuto, scarica tutto il necessario in automatico):
   ```bash
   docker compose build
   ```

Fatto: il software è pronto, non devi installare Python, PyTorch o altro a mano — è tutto già dentro l'immagine Docker.

### 👩‍💻 Versione sviluppatori

Due percorsi possibili:

**A. Container (consigliato, riproducibile):**
```bash
git clone git@github.com:giuseppedambruoso/eqnn.git && cd eqnn
cp .env.example .env   # imposta WANDB_API_KEY
docker compose build
```
Immagine multi-stage ([Dockerfile](Dockerfile)): stage `builder` con Poetry + toolchain di compilazione, stage `runtime` slim, utente non-root, dipendenze pinnate via `poetry.lock` (committato per build riproducibili).

**B. Ambiente locale con Poetry:**
```bash
poetry install
poetry shell
```
Richiede Python `>=3.11,<3.13` (vedi `pyproject.toml`). Per installare un pacchetto nuovo aggiornando il lock file: `poetry add <package>`; per installarlo solo nel venv senza toccare `pyproject.toml`: `pip install <package>` dentro `poetry shell`.

---

## 2. Eseguire un run di training

### 🔰 Guida per chi parte da zero

Nel terminale, dentro la cartella `eqnn`, esegui:
```bash
./run_interactive.sh
```
Lo script ti farà alcune domande in italiano, una alla volta (es. "Equivarianza [True]:"). Per ognuna:
- se premi solo **invio**, viene usato il valore tra parentesi quadre (il default, va bene nella maggior parte dei casi),
- altrimenti scrivi il valore che vuoi e premi invio.

Le domande principali:
- **Equivarianza / Twirling / Rimuovi cross-edge**: sono opzioni tecniche del modello quantistico, scrivi `True` o `False`. Se non sai cosa scegliere, lascia il default.
- **Numero immagini training (N)**: quante immagini usare per l'addestramento — un numero più basso (es. 20) fa un test veloce, un numero più alto (es. 300) allena meglio ma richiede più tempo.
- **Epoche**: quante volte il modello "ripassa" tutti i dati — più epoche possono migliorare il risultato ma allungano il tempo di attesa.
- **Seed**: un numero a caso per rendere l'esperimento ripetibile — lascialo pure com'è.

Alla fine ti verrà mostrato il comando che sta per essere eseguito e ti chiederà conferma (`Confermi? [Y/n]`): premi invio per procedere.

### 👩‍💻 Versione sviluppatori

Il progetto usa [Hydra](https://hydra.cc/) per la configurazione ([src/config/config.yaml](src/config/config.yaml)). `run_interactive.sh` è solo un wrapper che costruisce ed esegue `docker compose run` con gli override; puoi bypassarlo e scrivere gli override direttamente:

```bash
# run singolo
docker compose run --rm eqnn QNN.equivariance=False DATA.N=200 TRAINING.epochs=50

# sweep/grid (multirun Hydra, un run per combinazione)
docker compose run --rm eqnn -m QNN.equivariance=True,False QNN.reps=1,2,3

# raggruppare i run di uno sweep su wandb
docker compose run --rm -e WANDB_RUN_GROUP=exp-equiv-vs-no eqnn -m QNN.equivariance=True,False
```

Parametri configurabili: `GENERAL.{seed,dev}`, `DATA.{N,dataset,img_size,augment_test}`, `QNN.{num_qubits,p_err,reps,equivariance,twirling,remove_cross_edge,device}`, `TRAINING.{epochs,learning_rate,patience,min_delta}` — vedi [src/config/config.yaml](src/config/config.yaml) per i default. Per uno sweep strutturato su più parametri contemporaneamente vedi anche [run_experiments.sh](run_experiments.sh) (usa il launcher `joblib` per parallelizzare).

Per disattivare wandb (es. in CI o debug locale): `WANDB_MODE=disabled` in `.env`, oppure `-e WANDB_MODE=disabled` sulla riga di comando.

---

## 3. Monitorare l'avanzamento

### 🔰 Guida per chi parte da zero

Mentre il run è in corso, nel terminale vedrai una riga che si aggiorna da sola, tipo:
```
Job (Eq=True, Cross=True):  12%|████            | 12/100 [00:45<05:30, 3.7s/it]
```
Quel `12%` è la percentuale di avanzamento, e il tempo tra parentesi è quanto è già passato e quanto manca circa alla fine.

Poco prima, il terminale avrà stampato un link che inizia con `https://wandb.ai/...runs/...` — clicca (o copia e incolla nel browser) per aprire una pagina web che mostra in tempo reale i grafici dell'addestramento (quanto sta "imparando" il modello), aggiornati automaticamente mentre il run prosegue.

### 👩‍💻 Versione sviluppatori

- **Locale**: barra `tqdm` su stderr (posizionata per job quando si usa il launcher `joblib` in parallelo).
- **Remoto**: ogni run stampa due link all'avvio ([train.py](src/train.py:88)):
  - `View project` → https://wandb.ai/<entity>/eqnn — tabella di tutti i run, filtrabile/ordinabile per ogni campo di `config` (equivariance, N, seed, reps, ...), utile per confrontare esperimenti diversi.
  - `View run` → pagina del run corrente, con `train/loss`, `train/accuracy` aggiornati step-by-step, e a fine training anche le metriche di validazione e l'immagine della loss curve.
- Se il container gira in background (`docker compose run -d`), puoi seguirne i log con `docker logs -f <container_id>` (`docker ps` per trovarlo).

---

## 4. Vedere l'output e capire dove si salva

### 🔰 Guida per chi parte da zero

Quando il run finisce, i risultati restano salvati **sul tuo computer**, dentro la cartella del progetto, in `outputs/`. Dentro trovi una sottocartella con la data e l'ora del run (es. `outputs/2026-07-30/09-30-11/`), che contiene:
- `loss_history.jpg` — un grafico dell'andamento dell'errore durante l'addestramento (apribile con un doppio click come una normale immagine),
- `final_model.pt` — il modello addestrato salvato (un file tecnico, serve per riusare i risultati nel software, non è pensato per essere aperto direttamente),
- `main.log` — un file di testo con il resoconto testuale di cosa è successo durante il run.

In alternativa, tutti questi risultati (grafici, metriche, e il modello) sono visibili anche comodamente sul sito https://wandb.ai — nella pagina del tuo progetto "eqnn", senza dover cercare file sul computer.

### 👩‍💻 Versione sviluppatori

- Gli output Hydra vanno in `outputs/<date>/<time>/` per un run singolo, `multirun/<date>/<time>/<override_dirname>/` per uno sweep (`sweep.subdir` in `config.yaml`) — entrambe le cartelle sono montate come volumi Docker (vedi `docker-compose.yml`), quindi persistono sull'host anche se il container viene distrutto.
- Ogni run directory contiene ([train.py](src/train.py:171)): `loss_history.csv` (loss per epoca, raw), `loss_history.jpg` (plot), `final_model.pt` (`state_dict`-like: `{'params': tensor, 'val_acc': float}`, caricabile con `torch.load`), `main.log`, `.hydra/{config,overrides,hydra}.yaml` (config esatta usata, per riprodurre il run).
- Su wandb: stessa `loss_history.jpg` loggata come `wandb.Image`, più il modello versionato come **Artifact** (`model-<run_id>`, tipo `model`, tab "Artifacts" della run) — scaricabile e ricaricabile con l'API wandb, utile per non dipendere dal filesystem locale.
- La cache di MNIST è in `./data` (montata come volume) — scaricata una sola volta, riusata dai run successivi.

---

## 5. Sviluppo

Per contribuire codice (non necessario solo per lanciare esperimenti):

```bash
poetry install --with dev
poetry run pytest tests/ --disable-warnings --cov=src
poetry run ruff check .
poetry run black --check .
```

La CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) esegue automaticamente lint, test e build dell'immagine Docker ad ogni push/PR.
