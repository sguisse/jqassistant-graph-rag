#!/usr/bin/env bash

###
# jqassistant-graph-rag.sh — helper menu for building/running graphRAG 🚀
# Usage:
#   Interactive: ./jqassistant-graph-rag.sh
#   Direct:      ./jqassistant-graph-rag.sh --option:"01"
#
# Important environment variables:
#   Neo4j: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (or pass via CLI args --uri, --user, --password)
#   OpenAI: OPENAI_API_KEY, OPENAI_MODEL
#   DeepSeek: DEEPSEEK_API_KEY, DEEPSEEK_MODEL
#   Ollama: OLLAMA_BASE_URL, OLLAMA_MODEL
#   CLI LLM: LLM_CLI_CMD, LLM_CLI_PARAMS, LLM_CLI_TIMEOUT
#
# Cache & logs:
#   Summaries cached at: <project_path>/.cache/summary_cache.json
#   Default log file: debug.log (or set --log-file)
#
# Quick troubleshooting:
# - If Neo4j connection fails, confirm Bolt URI and port. jQAssistant embedded example uses bolt 7688; default in code is 7687.
# - If sentence-transformers fails to import, install torch appropriate for your system.
# - For LLM issues verify env vars; Ollama expects a local server.
#
# Where to look in the repo:
#   CLI entry: main.py
#   Arguments & defaults: input_params.py
#   LLM & embedding clients: llm_client.py
#   MCP tool server: mcp_server.py
#   Example ADK agent: rag_adk_agent
###

# Colors & emoji
RESET="\e[0m"
BOLD="\e[1m"
FG_BLUE="\e[34m"
BG_BLUE="\e[44m"
FG_WHITE="\e[97m"
BG_WHITE="\e[107m"
FG_CYAN="\e[36m"
FG_GREEN="\e[32m"
FG_YELLOW="\e[33m"
FG_MAGENTA="\e[35m"
FG_RED="\e[31m"

# Print header
print_header() {
  echo -e "${BOLD}${FG_MAGENTA}jqassistant-graph-rag helper${RESET}   ${FG_YELLOW}🧭${RESET}"
  echo
  echo -e "${BOLD}Quick actions${RESET}: choose an option (01-12) or pass --option:XX"
  echo
}

# Helper: generate a separator line for given column widths (accounts for surrounding spaces).
# Usage: _table_sep W1 W2 ...  →  +--W1+2--+--W2+2--+...
_table_sep() {
  local line="+"
  for w in "$@"; do
    line+=$(printf '%*s' $((w+2)) '' | tr ' ' '-')
    line+="+"
  done
  printf '%s\n' "$line"
}

# Global row registries (reset inside print_menu on each call)
_ROWS=()
_RCOLORS=()

# Register a table row.
# Usage: addRow "col1|col2|col3" [optional_FG_color_var]
# Example: addRow "01|Create venv|" "$FG_GREEN"
# If no color given, ODD rows use FG_CYAN and EVEN rows use FG_WHITE.
addRow() {
  _ROWS+=("$1")
  _RCOLORS+=("${2:-}")
}

# Print a neat ASCII table with alternating row colors (ODD=CYAN, EVEN=WHITE).
# Individual rows can override their color via addRow's second argument.
print_menu() {
  local C1=4 C2=54 C3=70

  # Separator is built dynamically: each column gets width+2 dashes (for the two border spaces)
  local sep
  sep=$(_table_sep $C1 $C2 $C3)

  # Header — pad content first, then wrap in BOLD so escape codes don't skew column math
  local h1 h2 h3
  h1=$(printf "%-${C1}s" "No")
  h2=$(printf "%-${C2}s" "Action")
  h3=$(printf "%-${C3}s" "Command / Notes")
  echo -e "${BOLD}${sep}${RESET}"
  echo -e "| ${BOLD}${h1}${RESET} | ${BOLD}${h2}${RESET} | ${BOLD}${h3}${RESET} |"
  echo -e "${BOLD}${sep}${RESET}"

  # Register rows (reset first so print_menu is idempotent)
  _ROWS=()
  _RCOLORS=()
  addRow "01|Create virtual environment and install Python deps|"
  addRow "02|Run enrichment only (no LLM calls)|Neo4j: bolt://localhost:7688 user \"\" / \"\""
  addRow "03|Run enrichment only (alternate)|Neo4j: bolt://localhost:7687 user neo4j / neo4j (default)"
  addRow "04|Run enrichment + generate summaries (fake LLM)|python3 main.py --generate-summary --llm-api fake"
  addRow "05|Run enrichment + summaries (OpenAI)|OPENAI_MODEL=\"gpt-4o\""
  addRow "06|Run enrichment + summaries (Ollama)|OLLAMA_BASE_URL=\"http://localhost:11434\"; OLLAMA_MODEL=\"your-model\""
  addRow "07|Run enrichment + summaries (CLI LLM: Gemini)|LLM_CLI_PARAMS=\"\"" "$FG_GREEN"
  addRow "08|Run enrichment + summaries (CLI LLM: Copilot)|LLM_CLI_PARAMS=\"--model gpt-5-mini --effort medium\"" "$FG_GREEN"
  addRow "09|Start embedded Neo4j server (Bolt: 7688, stays running)|tail -f /dev/null | mvn jqassistant:server &"
  addRow "10|Run the example ADK agent (CLI)|query \"Summarize the project\""
  addRow "11|Use the ADK web UI|adk web"
  addRow "12|Run the ADK agent directly|adk run rag_adk_agent"
  addRow "13|Check graph: Java analysis (.class, .java, labels)|zz-check-graph.py --mode java" "$FG_YELLOW"
  addRow "14|Check graph: config (APOC, schema, counts, embeddings)|zz-check-graph.py --mode config" "$FG_YELLOW"
  addRow "15|Start MCP server in background |(port ${MCP_PORT}, log → /tmp/mcp-server.log" "$FG_MAGENTA"
  addRow "16|List active server ports (Neo4j Bolt/HTTP, MCP)|nc + lsof check" "$FG_MAGENTA"
  addRow "00|Exit the helper script|" "$FG_RED"

  # Print rows — pad each cell before applying color so ANSI codes never offset column math
  local i
  for i in "${!_ROWS[@]}"; do
    local row="${_ROWS[$i]}"
    local forced="${_RCOLORS[$i]}"
    local color
    if [[ -n "$forced" ]]; then
      color="$forced"
    elif (( (i+1) % 2 == 0 )); then
      color="$FG_WHITE"
    else
      color="$FG_CYAN"
    fi
    IFS="|" read -r no title cmd _ <<< "$row"
    local p1 p2 p3
    p1=$(printf "%-${C1}s" "$no")
    p2=$(printf "%-${C2}s" "$title")
    p3=$(printf "%-${C3}s" "$cmd")
    echo -e "${color}| ${p1} | ${p2} | ${p3} |${RESET}"
  done

  echo -e "${BOLD}${sep}${RESET}"
}

# Execute low-level helper to run a command (shows before running)
run_cmd() {
  local cmd="$1"
  echo -e "${FG_GREEN}▶ Running:${RESET} ${BOLD}${cmd}${RESET}"
  bash -c "$cmd"
}

NEO4J_BOLT_URI="bolt://localhost:7688"
NEO4J_PARAMS="--uri ${NEO4J_BOLT_URI} --user '' --password ''"
NEO4J_MVN_PARAMS="-Djqassistant.store.embedded.connector-enabled=true -Djqassistant.store.embedded.bolt-port=7688"
PYTHON3_CMD=".venv/bin/python"
MCP_PORT=8800
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start the embedded jqassistant Neo4j server in the background.
# Uses 'tail -f /dev/null' to keep stdin open — jqassistant:server waits
# for stdin to close before exiting, so without this it dies immediately.
_start_neo4j_server() {
  local REPO_ROOT
  REPO_ROOT="$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || pwd)"
  echo -e "${FG_CYAN}▶ Starting embedded Neo4j server (Bolt: 7688) …${RESET}"
  tail -f /dev/null | mvn -f "${REPO_ROOT}/pom.xml" -Pjqassistant \
    ${NEO4J_MVN_PARAMS} \
    jqassistant:server > /tmp/jqa-server.log 2>&1 &
  local SERVER_PID=$!
  disown "$SERVER_PID"
  # Wait up to 30 s for Bolt port
  local i
  for i in $(seq 1 30); do
    nc -z localhost 7688 2>/dev/null && \
      echo -e "  ${FG_GREEN}✅  Neo4j Bolt ready on port 7688 (PID $SERVER_PID)${RESET}" && return 0
    sleep 1
  done
  echo -e "  ${FG_RED}❌  Bolt port 7688 did not open in 30 s — check /tmp/jqa-server.log${RESET}"
  return 1
}

# Start the MCP server (FastMCP streamable-http) in the background.
_start_mcp_server() {
  # Resolve repository root and ensure we use the venv python if present.
  # Script directory (where mcp_server.py lives) — prefer this over repo root
  local SCRIPT_DIR
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  echo -e "${FG_CYAN}▶ Starting MCP server (port ${MCP_PORT}) in ${SCRIPT_DIR} …${RESET}"

  # Resolve Python interpreter: prefer configured PYTHON3_CMD (relative to script dir),
  # then .venv in script dir, then system python3/python.
  local PY_BIN
  # If PYTHON3_CMD is absolute use it directly; otherwise resolve relative to SCRIPT_DIR
  if [[ "$PYTHON3_CMD" == /* ]]; then
    PY_CAND="$PYTHON3_CMD"
  else
    PY_CAND="$SCRIPT_DIR/$PYTHON3_CMD"
  fi
  if [[ -x "$PY_CAND" ]]; then
    PY_BIN="$PY_CAND"
  elif [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PY_BIN="$SCRIPT_DIR/.venv/bin/python"
  else
    PY_BIN="$(command -v python3 || command -v python || true)"
  fi

  if [[ -z "$PY_BIN" ]]; then
    echo -e "${FG_RED}❌  No python interpreter found (checked PYTHON3_CMD, .venv/bin/python, system python3).${RESET}"
    return 2
  fi

  # Check for required Python dependency (fastmcp) before attempting start.
  # Capture stderr so we can show import-time errors (pydantic / python version incompatibilities).
  TMP_ERR=/tmp/mcp-import.err
  rm -f "$TMP_ERR" 2>/dev/null || true
  "$PY_BIN" -c "import fastmcp; print('fastmcp_import_ok')" 2>"$TMP_ERR"
  if [[ $? -ne 0 ]]; then
    echo -e "${FG_RED}❌  Failed to import 'fastmcp' using interpreter: $PY_BIN${RESET}"
    echo -e "  Error output from interpreter (first 40 lines):"
    sed -n '1,40p' "$TMP_ERR" || true
    echo
    echo -e "  Possible fixes:"
    echo -e "    • Recreate the tool venv with a compatible Python (example: python3.12 -m venv .venv)"
    echo -e "    • Or run option 01 to install requirements into the tool venv and ensure the venv Python is compatible."
    echo -e "  After fixing, re-run this option to start the MCP server."
    return 3
  fi

  # If a local sentence-transformer model exists, point to it to avoid network downloads
  # (avoids SSL cert failures on corporate networks). Falls back to default "all-MiniLM-L6-v2"
  # which requires HuggingFace download if not already cached.
  local LOCAL_MODEL="$SCRIPT_DIR/models/all-MiniLM-L6-v2"
  if [[ -d "$LOCAL_MODEL" ]]; then
    export SENTENCE_TRANSFORMER_MODEL="$LOCAL_MODEL"
    echo -e "  ${FG_CYAN}Using local model: ${LOCAL_MODEL}${RESET}"
  else
    echo -e "  ${FG_YELLOW}⚠️  Local model not found at ${LOCAL_MODEL}.${RESET}"
    echo -e "     If HuggingFace is not reachable (SSL/corporate network),"
    echo -e "     download the model first: huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --local-dir ${LOCAL_MODEL}"
    echo -e "     Or set SSL_CERT_FILE: export SSL_CERT_FILE=\$($PY_BIN -m certifi)"
    # Set SSL cert to certifi bundle to improve chances of download succeeding
    SSL_CERT="$($PY_BIN -m certifi 2>/dev/null)"
    [[ -n "$SSL_CERT" ]] && export SSL_CERT_FILE="$SSL_CERT" && export REQUESTS_CA_BUNDLE="$SSL_CERT"
  fi

  # Launch from script directory so mcp_server.py relative imports work and logs are predictable
  (cd "$SCRIPT_DIR" || exit 1
    nohup "$PY_BIN" mcp_server.py ${NEO4J_PARAMS} > /tmp/mcp-server.log 2>&1 &
    MCP_PID=$!
    # Persist PID for later inspection
    echo "$MCP_PID" > /tmp/mcp-server.pid
    disown "$MCP_PID"
  )

  # Wait up to 15s for the port to open
  local i
  for i in $(seq 1 15); do
    if nc -z localhost "$MCP_PORT" 2>/dev/null; then
      echo -e "  ${FG_GREEN}✅  MCP server ready on port ${MCP_PORT} (PID $(< /tmp/mcp-server.pid))${RESET}"
      echo -e "  Logs: /tmp/mcp-server.log (PID file: /tmp/mcp-server.pid)"
      return 0
    fi
    sleep 1
  done

  echo -e "  ${FG_RED}❌  MCP port ${MCP_PORT} did not open in 15 s — check /tmp/mcp-server.log and /tmp/mcp-server.pid${RESET}"
  return 1
}

# Print a summary of known server ports and whether they are reachable.
_list_ports() {
  local ports=(
    "Neo4j Bolt     7688"
    "Neo4j HTTP     7777"
    "MCP server     ${MCP_PORT}"
  )
  echo -e "\n${BOLD}Server port status${RESET}"
  printf '%-24s %-8s %s\n' "Service" "Port" "Status"
  printf '%-24s %-8s %s\n' "-------" "----" "------"
  for entry in "${ports[@]}"; do
    local svc port
    svc=$(echo "$entry" | awk '{print $1, $2}')
    port=$(echo "$entry" | awk '{print $3}')
    if nc -z localhost "$port" 2>/dev/null; then
      printf "${FG_GREEN}%-24s %-8s %s${RESET}\n" "$svc" "$port" "✅  OPEN"
    else
      printf "${FG_RED}%-24s %-8s %s${RESET}\n" "$svc" "$port" "❌  closed"
    fi
  done
  echo
  echo -e "${FG_CYAN}Active listeners (lsof):${RESET}"
  lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | awk 'NR==1 || /:(7688|7777|'"$MCP_PORT"')/' || true
  echo
}

# Ensure Neo4j is reachable; start it automatically if not.
_ensure_neo4j() {
  if nc -z localhost 7688 2>/dev/null; then
    echo -e "  ${FG_GREEN}✅  Neo4j already reachable on port 7688${RESET}"
    return 0
  fi
  echo -e "  ${FG_YELLOW}⚠️   Neo4j not reachable — starting embedded server …${RESET}"
  _start_neo4j_server
}

# Option handlers
do_option() {
  local opt="$1"
  case "$opt" in
    0|00)
      echo -e "${FG_RED}Exiting...${RESET} 👋"
      exit 0
      ;;
    1|01)
      # Create venv and install requirements in the tool's script directory (SCRIPT_DIR)
      # Prefer python3.12 for venv creation if available (pydantic/fastmcp are sensitive to py version);
      # fallback to system python3.
      run_cmd "cd \"${SCRIPT_DIR}\" && PY_CREATOR=\$(command -v python3.12 || command -v python3 || command -v python) && echo 'Using venv creator:' \$PY_CREATOR && \"\$PY_CREATOR\" -m venv .venv && \"${SCRIPT_DIR}/.venv/bin/python\" -m pip install --upgrade pip && \"${SCRIPT_DIR}/.venv/bin/pip\" install -r requirements.txt"
      ;;
    2|02)
      run_cmd "$PYTHON3_CMD main.py $NEO4J_PARAMS"
      ;;
    3|03)
      run_cmd "$PYTHON3_CMD main.py"
      ;;
    4|04)
      run_cmd "$PYTHON3_CMD main.py --generate-summary --llm-api fake $NEO4J_PARAMS"
      ;;
    5|05)
      echo -e "${FG_YELLOW}⚠️  Ensure OPENAI_API_KEY is set before running.${RESET}"
      read -p "Set and export env vars now? (y/n) " yn
      if [[ "$yn" =~ ^[Yy] ]]; then
        read -p "OPENAI_API_KEY: " key
        read -p "OPENAI_MODEL (optional, default gpt-3.5-turbo): " model
        export OPENAI_API_KEY="$key"
        [[ -n "$model" ]] && export OPENAI_MODEL="$model"
      fi
      run_cmd "$PYTHON3_CMD main.py --generate-summary --llm-api openai $NEO4J_PARAMS"
      ;;
    6|06)
      echo -e "${FG_YELLOW}⚠️  Ensure Ollama server is reachable.${RESET}"
      read -p "OLLAMA_BASE_URL (default http://localhost:11434): " ob
      read -p "OLLAMA_MODEL (e.g. your-model): " om
      export OLLAMA_BASE_URL="${ob:-http://localhost:11434}"
      [[ -n "$om" ]] && export OLLAMA_MODEL="$om"
      run_cmd "$PYTHON3_CMD main.py --generate-summary --llm-api ollama $NEO4J_PARAMS"
      ;;
    7|07)
      export SENTENCE_TRANSFORMER_MODEL=$(pwd)/models/all-MiniLM-L6-v2
      export SSL_CERT_FILE=$(python -m certifi)
      export REQUESTS_CA_BUNDLE=$(python -m certifi)
      export LLM_CLI_CMD="${LLM_CLI_CMD:-gemini}"
      export LLM_CLI_PARAMS="${LLM_CLI_PARAMS:-}"
      export LLM_CLI_TIMEOUT="${LLM_CLI_TIMEOUT:-300}"
      run_cmd "$PYTHON3_CMD main.py --generate-summary --llm-api cli $NEO4J_PARAMS"
      ;;
    8|08)
      export LLM_CLI_CMD="${LLM_CLI_CMD:-copilot}"
      export LLM_CLI_PARAMS="${LLM_CLI_PARAMS:---model gpt-5-mini --effort medium}"
      export LLM_CLI_TIMEOUT="${LLM_CLI_TIMEOUT:-300}"
      run_cmd "$PYTHON3_CMD main.py --generate-summary --llm-api cli $NEO4J_PARAMS"
      ;;
    9|09)
      _start_neo4j_server
      ;;
    10)
      run_cmd "$PYTHON3_CMD rag_adk_agent/run_agent.py --query \"Summarize the project\""
      ;;
    11)
      echo -e "${FG_YELLOW}Note: 'adk' CLI comes with google-adk package. Ensure it is installed.${RESET}"
      run_cmd "$PYTHON3_CMD -m adk web"
      ;;
    12)
      run_cmd "$PYTHON3_CMD -m adk run rag_adk_agent"
      ;;
    13)
      _ensure_neo4j && run_cmd "$PYTHON3_CMD zz-check-graph.py --mode java $NEO4J_PARAMS"
      ;;
    14)
      _ensure_neo4j && run_cmd "$PYTHON3_CMD zz-check-graph.py --mode config $NEO4J_PARAMS"
      ;;
    15)
      _start_mcp_server
      ;;
    16)
      _list_ports
      ;;
    *)
      echo -e "${FG_RED}Invalid option: $opt${RESET}"
      ;;
  esac
}

# Parse args: support --option:XX or --option=XX or -o XX
requested_option=""
for a in "$@"; do
  if [[ "$a" == --option:* ]]; then
    requested_option="${a#--option:}"
    requested_option="${requested_option%\"}"
    requested_option="${requested_option#\"}"
  elif [[ "$a" == --option=* ]]; then
    requested_option="${a#--option=}"
  elif [[ "$a" == -o ]]; then
    # next arg is option
    :
  fi
done

# Also handle -o 01 style
while getopts ":o:-:" optchar; do
  case "${optchar}" in
    o)
      requested_option="${OPTARG}"
      ;;
    -)
      # long options (if any)
      ;;
    \?)
      ;;
  esac
done

# Interactive if no option specified
if [[ -z "$requested_option" ]]; then
  print_header
  print_menu
  read -p $'\nSelect option (00-16, q to quit): ' selection
  if [[ "$selection" =~ ^[Qq] ]]; then
    echo "Bye 👋"
    exit 0
  fi
  do_option "$selection"
else
  # direct invocation
  # strip possible surrounding quotes
  requested_option="${requested_option%\"}"
  requested_option="${requested_option#\"}"
  # normalize leading zero removal for case statement compatibility
  do_option "$requested_option"
fi
