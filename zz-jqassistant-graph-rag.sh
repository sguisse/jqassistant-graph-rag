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
  addRow "09|Start the MCP tool server|Neo4j: bolt://localhost:7688 user neo4j / <pw>"
  addRow "10|Run the example ADK agent (CLI)|query \"Summarize the project\""
  addRow "11|Use the ADK web UI|adk web"
  addRow "12|Run the ADK agent directly|adk run rag_adk_agent"
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

NEO4J_PARAMS="--uri bolt://localhost:7688 --user '' --password ''"
PYTHON3_CMD=".venv/bin/python"

# Option handlers
do_option() {
  local opt="$1"
  case "$opt" in
    0|00)
      echo -e "${FG_RED}Exiting...${RESET} 👋"
      exit 0
      ;;
    1|01)
      run_cmd "python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
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
      run_cmd "$PYTHON3_CMD mcp_server.py $NEO4J_PARAMS"
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
  read -p $'\nSelect option (00-12, q to quit): ' selection
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
