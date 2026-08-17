# Install P2Predict

This guide assumes you have **never used a terminal, never installed Python, and don't know what an API is**. That's fine, you don't need to. Follow the steps for your computer and copy-paste each command exactly.

**What you're actually doing:** installing a small program on your laptop, then telling your AI agent assistant (Claude or others) that it exists. After that you never touch the terminal again — you just talk to Claude in plain English about your pricing data. You can actually just give your agent the URL to this page and it will install p2predict for you in a folder of your choosing.

**Time:** about 15 minutes

**Your data never leaves your machine.** P2Predict reads your spreadsheet, does its maths, and answers — all locally. Nothing is uploaded anywhere.

---

## Do I need admin rights?

I am assuming that most users are managers and analysts in large global companies, so I wrote this install guide for you. Most business machines are locked and admin mode is not easy to get. If you have windows, you dont need admin mode - however on a mac, you do need it to install a modern version of python.

| | Admin password needed? |
|---|---|
| **Windows** | **No.** Python installs "just for you", into your own user folder. |
| **Mac** | **Once**, to install Python itself. Everything after that is admin-free. |

Why the Mac needs it: macOS ships with Python 3.9, which is too old for P2Predict (it needs 3.10 or newer), so you have to install a current version, and Apple's installer asks for an admin password.

If you don't have an admin password on your Mac, ask IT to install **Python 3.12 from python.org** — this is a standard, signed installer. Then start at step 2.

> **A note on jargon.** *PowerShell* (Windows) / *Terminal* (Mac) is a window where you type commands instead of clicking. *Python* is the language P2Predict is written in — you'll never write any code. *MCP* is just the plug that lets Claude or an AI agent talk to programs on your computer.

---

# Windows: step by step

## Step 1 — Install Python

1. Go to **[python.org/downloads](https://www.python.org/downloads/)**.
2. Click the big yellow **Download Python 3.12.x** button. (3.12 or 3.13 are the safest choices. Avoid the newest release for a few months after it comes out — some components lag behind.)
3. Open the downloaded `.exe`.
4. At the bottom of the first window, tick the box that says **"Add python.exe to PATH"** *before* clicking anything else.

   *PATH is just a list of folders Windows searches when you type a command. If Python isn't on it, Windows says "python is not recognized" even though Python is sitting right there.*
5. Click **Install Now** (not "Customize"). This installs into your own user folder and does **not** need an admin password.
6. If it offers **"Disable path length limit"** at the end, click it. Then click **Close**.

## Step 2 — Open PowerShell

Press the **Windows key**, type `PowerShell`, press **Enter**. Use the normal one — you do *not* need "Run as administrator".

To run a command: copy it, right-click inside the PowerShell window (that pastes), press **Enter**, and wait for a fresh line.

Check Python arrived:

```powershell
python --version
```

You want `Python 3.12.x` (or any number 3.10 and above). If nothing happens, or the Microsoft Store opens, see [Troubleshooting](#troubleshooting).

## Step 3 — Create a home for P2Predict

This makes a `P2Predict` folder in your user folder with a self-contained Python setup inside. It can't affect anything else on your PC, and deleting the folder removes it completely.

```powershell
mkdir $HOME\P2Predict\models
```

```powershell
python -m venv $HOME\P2Predict\venv
```

The second command takes a few seconds and prints nothing. Silence means success.

## Step 4 — Install P2Predict

```powershell
cd $HOME\P2Predict
```

```powershell
.\venv\Scripts\pip.exe install "p2predict[mcp]"
```

This takes 1–3 minutes and prints a lot of scrolling text. The last line should say `Successfully installed ...` with a long list.

Check it worked:

```powershell
.\venv\Scripts\p2predict.exe --version
```

If it prints `p2predict 0.9.6` (or a higher number), you're done installing. Now jump to [**Connect it to Claude**](#connect-it-to-claude).

---

# Mac: step by step

## Step 1 — Install Python

1. Go to **[python.org/downloads](https://www.python.org/downloads/)**.
2. Click the big yellow **Download Python 3.12.x** button. (3.12 or 3.13 are the safest choices. Avoid the newest release for a few months after it comes out — some components lag behind.)
3. Open the downloaded `.pkg` file and click **Continue → Continue → Agree → Install**.
4. Enter your Mac password when asked.
5. When it finishes, a Finder window may pop open. You can close it.

## Step 2 — Open Terminal

Press **Cmd + Space**, type `Terminal`, press **Enter**.

A window with text appears. This is where the next commands go. To run a command: copy it, click into the Terminal window, press **Cmd + V**, press **Enter**, and wait until the text stops scrolling and you get a fresh line.

Check Python arrived:

```bash
python3 --version
```

You want `Python 3.12.x` (or any number 3.10 and above). If you get an error, see [Troubleshooting](#troubleshooting).

## Step 3 — Create a home for P2Predict

This makes a folder called `P2Predict` in your home folder and puts a self-contained Python setup inside it. Self-contained means it can't break anything else on your Mac, and deleting the folder removes it completely.

```bash
mkdir -p ~/P2Predict/models
```

```bash
python3 -m venv ~/P2Predict/venv
```

The second command takes a few seconds and prints nothing. Silence means success.

## Step 4 — Install P2Predict

```bash
~/P2Predict/venv/bin/pip install "p2predict[mcp]"
```

This downloads P2Predict and everything it needs. It takes 1–3 minutes and prints a lot of scrolling text. The last line should say `Successfully installed ...` with a long list.

> **Why the quotation marks?** Without them the Mac terminal treats `[mcp]` as a wildcard and fails with `zsh: no matches found`. The quotes are required — don't drop them.

## Step 5 — Install the XGBoost helper (Mac only)

P2Predict's most accurate model, **XGBoost**, needs one small shared maths library on Mac called **libomp**. It doesn't come with Python, and without it P2Predict won't start. (Windows and Linux include it automatically — this step is Mac-only, but on Mac it is not optional.)

libomp is installed with **Homebrew**, the standard Mac tool for this kind of thing. First check whether you already have Homebrew:

```bash
brew --version
```

- **If it prints a version number**, you have Homebrew. Skip straight to the `brew install libomp` command below.
- **If it says `command not found`**, install Homebrew first. Paste this, press Enter, and follow its prompts (it will ask for your Mac password, and takes a few minutes):

  ```bash
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  ```

  When it finishes, **close Terminal completely (Cmd + Q) and reopen it** — Homebrew isn't on your PATH until you do.

Now install the helper. This is one command, once:

```bash
brew install libomp
```

Check everything worked:

```bash
~/P2Predict/venv/bin/p2predict --version
```

If it prints `p2predict 0.9.6` (or a higher number), you're done installing. Now jump to [**Connect it to Claude**](#connect-it-to-claude).

---

# Connect it to Claude (example here is claude. For copilot and other agentic models, refer to the vendor as MCP is standard format)

Claude needs to be told where P2Predict lives. You'll paste a small block of settings into one file.

### Get your settings block

P2Predict can write this for you. Run the command for your computer, and keep the window open.

**Windows:**

```powershell
& "$HOME\P2Predict\venv\Scripts\p2predict-mcp.exe" --models-dir "$HOME\P2Predict\models" --print-config
```

**Mac:**

```bash
~/P2Predict/venv/bin/p2predict-mcp --models-dir ~/P2Predict/models --print-config
```

It prints the exact block to paste, with your real folder locations already filled in and everything spelled correctly for your machine. Copy everything from the first `{` to the last `}`.

This is worth using rather than typing the paths yourself. Getting them slightly wrong is the most common reason Claude doesn't find P2Predict, and the mistakes are invisible: a folder name that doesn't match, or a Windows path that needs every `\` written twice.

### Open Claude's config file

In **Claude Desktop**: menu **Claude → Settings → Developer → Edit Config**. That opens the right file directly — much easier than hunting for it.

If you need the file manually:
- **Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
- **Mac:** `~/Library/Application Support/Claude/claude_desktop_config.json`

### Paste the settings

If the file is empty or brand new, paste in the block you just copied. It looks like this, but with your real folders instead of `YOURNAME`:

```json
{
  "mcpServers": {
    "p2predict": {
      "command": "C:\\Users\\YOURNAME\\P2Predict\\venv\\Scripts\\p2predict-mcp.exe",
      "args": ["--models-dir", "C:\\Users\\YOURNAME\\P2Predict\\models"]
    }
  }
}
```

(On Mac the `command` is a path like `/Users/YOURNAME/P2Predict/venv/bin/p2predict-mcp` instead — but let `--print-config` fill it in either way.)

**If the file already has something in it**, don't replace it. Add just the `"p2predict": { ... }` part alongside whatever is already inside `"mcpServers"`, separated by a comma.

Save the file, then **quit Claude Desktop completely and reopen it**. On Mac that means **Cmd + Q**, not just closing the window.

### Using Claude Code or Cursor instead?

Same server, different way to register it. For Claude Code, run this once (with your real path):

```bash
claude mcp add p2predict -- ~/P2Predict/venv/bin/p2predict-mcp --models-dir ~/P2Predict/models
```

For Cursor and other clients, add the same JSON block through that client's own MCP settings screen.

### Check it connected

In Claude Desktop, look for the tools/plug icon near the message box — `p2predict` should be listed. Or just ask:

> *"Do you have P2Predict available? List any models you can see."*

It should answer that it's connected and that there are no models yet. That's the correct answer on a fresh install — you haven't trained one.

---

# Your first conversation

You need a spreadsheet saved as **CSV** with one row per part: some columns describing the part, and one column with the price you paid.

```csv
Part,Weight,Region,Supplier,Size,Price
CP17-17921595,17,EU,supplier A,Standard,1.41
CP2-5580430,2,CN,supplier A,Small,0.18
CP30-19674030,30,SG,supplier A,Large,2.15
```

In Excel: **File → Save As → CSV**. Aim for at least a few hundred rows. Now just talk to Claude:

> *"Train a pricing model on my file at ~/P2Predict/parts.csv, predicting Price."*
>
> *"What should a 25 kg part from Supplier A in the EU cost?"*
>
> *"Supplier B quoted me $4.10 for that. Is that fair?"*
>
> *"What happens to the price if we switch to Supplier B?"*
>
> *"Benchmark these 200 RFQ line items against the model."*

Claude will look at your data, tell you in plain language what it found, and ask before training. You don't need to know which algorithm it picked — ask *"how much should I trust this?"* and it will tell you honestly where the model is reliable and where it isn't.

---

## What the agent can do

Claude picks the right tool for your question automatically. You never call these by name.

| Tool | What it does |
|---|---|
| `list_models` | Discover trained models |
| `get_model_info` | Features, types, categories, calibration status |
| `get_model_quality` | Trust verdict: is the model reliable, where, and by how much |
| `predict` | Point estimate for a single part |
| `predict_batch` | Predict multiple parts in one call |
| `predict_from_csv` | Batch-predict a CSV file |
| `predict_interval` | The likely range (e.g. 9-in-10 coverage) |
| `explain` | Per-feature attribution — what drives this price |
| `what_if` | What changes if I switch supplier / material / spec? |
| `propose_training_plan` | Review the data and propose how to train, before training |
| `train` | Train a new model from a CSV |
| `generate_report` | Model-quality PDF report |

---

## Updating

**Mac:**

```bash
~/P2Predict/venv/bin/pip install --upgrade "p2predict[mcp]"
```

**Windows:**

```powershell
& "$HOME\P2Predict\venv\Scripts\pip.exe" install --upgrade "p2predict[mcp]"
```

(The leading `&` is PowerShell's way of running a program from a quoted path. It's required — don't drop it.)

Quit and reopen Claude Desktop afterwards so it picks up the new version. Your trained models in the `models` folder are untouched.

## Uninstalling

Delete the `P2Predict` folder from your home folder, and remove the `"p2predict"` block from Claude's config file. That's it — nothing else was touched. (Your trained models are inside that folder, so move them out first if you want to keep them.)

---

## Troubleshooting

Find your exact error message below.

### `zsh: no matches found: p2predict[mcp]` (Mac)

You left out the quotation marks. Use `"p2predict[mcp]"`, with the quotes.

### `error: externally-managed-environment` (Mac)

You ran `pip` directly instead of the one inside your P2Predict folder. Modern Macs block that on purpose. Use the full path — `~/P2Predict/venv/bin/pip` — exactly as written in step 4.

### `python is not recognized...` (Windows)

You missed the **"Add python.exe to PATH"** tickbox during install. Easiest fix: re-run the Python installer, choose **Modify**, click through to the last screen and make sure the PATH option is on. Or uninstall Python and reinstall, this time ticking the box.

### Typing `python` opens the Microsoft Store (Windows)

That's a Windows placeholder, not real Python. Install Python from [python.org](https://www.python.org/downloads/) with the PATH box ticked. If it still happens: **Settings → Apps → Advanced app settings → App execution aliases**, and turn off both `python.exe` and `python3.exe`.

### `SSLError`, `Could not fetch URL`, `Retrying...`, or `Connection timed out` during install (usually a work computer)

This is the **most common reason an install fails on a locked-down company laptop**, and it has nothing to do with P2Predict. Your company network sends all internet traffic through a security gateway (a "proxy"). The install command doesn't know that gateway exists, so it can't reach the internet to download anything.

How to tell that's what this is: try the exact same install command on a **different network** — your home wifi, or your phone's hotspot. If it works there, the company network is the cause.

To make it work on the company network, you'll need one of two things from your **IT department** (this is a normal request — copy them the line below):

> *"To install a Python tool, I need either the company's pip proxy setting, or the address of our internal PyPI mirror (Artifactory / Nexus). Which should I use?"*

- If they give you an **internal mirror address**, add it to the install command like this (Windows shown, with your real address):

  ```powershell
  .\venv\Scripts\pip.exe install --index-url https://your-company-mirror/simple "p2predict[mcp]"
  ```

- If they give you a **proxy address**, set it first, then re-run the normal install in the *same* window:

  ```powershell
  $env:HTTPS_PROXY = "http://proxy.yourcompany.com:8080"
  ```

### The install downloads for a while, then stops or times out

One of the pieces (XGBoost) is about 100 MB, and a slow or throttled work connection can drop the download partway. **Just run the exact same install command again** — it picks up where it left off. If it keeps stopping at the same place every time, it's the proxy issue directly above, not a bad connection.

### `ImportError: DLL load failed while importing ...` (Windows) — when Claude uses P2Predict, or when you run a command

P2Predict installed correctly, but one of its maths engines needs a small, standard Microsoft system library that some fresh or stripped-down Windows machines are missing. It's called the **Microsoft Visual C++ Redistributable** — think of it as a shared toolbox that lots of Windows programs rely on. (Most machines already have it, which is why this one is rare.)

Install it once from Microsoft's official page: **[aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe)** (a signed Microsoft installer). This one **may ask for an admin password** — if your machine is locked down, send that link to IT and ask them to install *"the latest Visual C++ x64 Redistributable."* Then quit and reopen Claude.

### `Library not loaded: @rpath/libomp.dylib` or `XGBoostError ... OpenMP runtime is not installed` (Mac)

P2Predict installed fine, but its XGBoost model can't find the **libomp** helper — you skipped [Step 5](#step-5--install-the-xgboost-helper-mac-only), or it didn't finish. Go back and run it:

```bash
brew install libomp
```

If `brew` itself says `command not found`, install Homebrew first — the two commands are in [Step 5](#step-5--install-the-xgboost-helper-mac-only). Then quit and reopen Claude.

### `command not found: python3` (Mac)

Python didn't install, or Terminal was open before you installed it. Quit Terminal completely (**Cmd + Q**) and reopen it, then try again. If it still fails, re-run the python.org installer.

### `ModuleNotFoundError: No module named 'mcp.server.fastmcp'`

You're on version 0.9.4 or older, which could pull in an incompatible component. [Update](#updating) to 0.9.5 or newer, which pins the working version.

### `ModuleNotFoundError: No module named 'mcp'`

You installed P2Predict without the `[mcp]` part. Re-run step 4 exactly as written.

### `p2predict-train` crashes with `numpy.core.multiarray failed to import`

An old version paired itself with an incompatible component. [Update](#updating) to 0.9.4 or newer.

### Claude doesn't show P2Predict in its tools

Work through these in order:

1. Did you **fully quit** Claude Desktop and reopen it? On Mac, **Cmd + Q** — closing the window isn't enough.
2. **Re-run the `--print-config` command from [Get your settings block](#get-your-settings-block) and paste its output over what's in the file.** That rules out every path and spelling mistake at once, which is what almost all of these turn out to be.
3. Is the JSON valid? A missing comma or bracket makes Claude ignore the whole file silently. Paste it into [jsonlint.com](https://jsonlint.com) to check — this is the one thing `--print-config` can't fix for you, because it happens when you merge its output with settings that were already there.
4. Does the server actually start? Run the `command` from your config in the terminal, followed by `--models-dir` and your models folder. If it prints `P2Predict MCP server v... loaded from ...` and then sits there waiting, that's correct — press **Ctrl + C** to stop it. If it errors instead, the error tells you what's wrong.

### The console commands work in Terminal but not elsewhere

That's a PATH issue, and the full-path approach in this guide avoids it entirely. If you installed a different way and want the short commands to work everywhere, the alternative is to call Python directly, which never depends on PATH:

```bash
python -m p2predict.mcp --models-dir /path/to/models
```

```bash
python -m p2predict.cli.train -i data.csv -t Price
```

Run `pip show -f p2predict` to see where the console scripts were actually placed.

---

## Keeping your data safe

- P2Predict reads your CSV, trains, and predicts entirely on your machine over a local connection. Nothing is uploaded.
- Don't commit raw vendor data or credentials to a shared repository — keep raw pulls and API keys outside version control.
- Respect the terms of service of any catalog or data source you pull from.

## Other surfaces

The same engine is available three ways, all calling the same maths:

- **AI agent (MCP)** — the primary interface, described above.
- **Command line** — `p2predict` and `p2predict-train` for scripted or interactive use.
- **Python API** — `from p2predict import auto_train, explain, predict_interval, what_if` for embedding in a notebook or pipeline.

Full reference for the CLI, Python API, JSON output schema, and CSV data format: **[TECHNICAL.md](TECHNICAL.md)**.

## Do I need to pay for this?

No. Any company — including a for-profit one — can use P2Predict internally for its own operations, procurement, and benchmarking at no cost. There's nothing to buy, no licence key, no trial period, and no one to call.

A paid licence is only needed if you use P2Predict to serve third-party clients (consulting or advisory work) or build it into a product you sell. Full terms in [LICENSE](LICENSE); commercial enquiries via [ahmedhafsi.com/contact](https://ahmedhafsi.com/contact/).

## Contributing

Bug reports, feature requests, and dataset suggestions — [open an issue](https://github.com/ahmed-khalil-hafsi/P2Predict/issues).

Open procurement datasets are especially welcome: ICs, passive components, plastic parts, mechanical parts. If you know of one, or your organization would share, [reach out](https://ahmedhafsi.com/contact/).

Code contributions require a CLA (P2Predict is dual-licensed). [Reach out](https://ahmedhafsi.com/contact/) before investing time in a large patch.
