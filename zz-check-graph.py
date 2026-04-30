#!/usr/bin/env python3
"""
zz-check-graph.py — Graph health checker for jqassistant-graph-rag
===================================================================
Two check modes (pass --mode as first argument):

  java     Check that Java bytecode and source files were properly scanned
           and that Spring stereotype labels were applied (option 13).

  config   Check graph configuration: APOC availability, label presence,
           node/relationship counts, enrichment state (option 14).

Usage
-----
  python3 zz-check-graph.py [--mode java|config] [--uri bolt://...] \\
                             [--user USER] [--password PWD]

Exit codes
----------
  0  All checks passed
  1  One or more checks failed
  2  Cannot connect to Neo4j
"""

import argparse
import logging
import sys
import warnings

# Silence the verbose Neo4j server notification messages that the driver
# emits via the standard logging system (UnknownLabel, UnknownProperty, etc.)
logging.getLogger("neo4j").setLevel(logging.ERROR)
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _ok(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {GREEN}✅  {label}{RESET}{suffix}")


def _warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {YELLOW}⚠️   {label}{RESET}{suffix}")


def _fail(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {RED}❌  {label}{RESET}{suffix}")


def _info(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  {CYAN}ℹ️   {label}{RESET}{suffix}")


def _header(title: str) -> None:
    print(f"\n{BOLD}{title}{RESET}")
    print("─" * (len(title) + 4))


def _count(session, cypher: str, param: str = "n") -> int:
    return session.run(cypher).single()[param]


# ── Neo4j connection ──────────────────────────────────────────────────────────


def connect(uri: str, user: str, password: str):
    try:
        from neo4j import GraphDatabase  # type: ignore

        # Suppress Neo4j server notifications (UnknownLabel, UnknownProperty) —
        # these appear for labels/properties that don't exist yet and are expected.
        try:
            from neo4j import warnings as neo4j_warnings  # type: ignore

            warnings.filterwarnings("ignore", category=neo4j_warnings.Neo4jWarning)
        except Exception:
            pass
    except ImportError:
        print(f"{RED}❌  neo4j Python driver not installed.{RESET}")
        print(
            "    Run:  pip install neo4j   (or use the venv: .venv/bin/pip install neo4j)"
        )
        sys.exit(2)

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        # validate connectivity
        with driver.session() as s:
            s.run("RETURN 1").single()
        return driver
    except Exception as exc:
        print(f"{RED}❌  Cannot connect to Neo4j at {uri}: {exc}{RESET}")
        sys.exit(2)


# ── Mode: java ────────────────────────────────────────────────────────────────


def check_java(session) -> bool:
    """Verify Java bytecode + source scan completeness and Spring label enrichment."""
    passed = True

    _header("🔍  Java Analysis Check")

    # 1. Bytecode: :Java:Type:Class
    n_class = _count(session, "MATCH (c:Java:Type:Class) RETURN count(c) AS n")
    if n_class > 0:
        _ok(":Java:Type:Class nodes present", f"{n_class} classes")
    else:
        _fail(
            ":Java:Type:Class = 0  →  bytecode not scanned",
            "run: mvn -Pjqassistant jqassistant:scan jqassistant:analyze",
        )
        passed = False

    # 2. :Method nodes
    n_method = _count(session, "MATCH (m:Method) RETURN count(m) AS n")
    if n_method > 0:
        _ok(":Method nodes present", f"{n_method}")
    else:
        _warn(":Method = 0  →  may indicate incomplete scan")

    # 3. :Package nodes
    n_pkg = _count(session, "MATCH (p:Package) RETURN count(p) AS n")
    if n_pkg > 0:
        _ok(":Package nodes present", f"{n_pkg}")
    else:
        _warn(":Package = 0")

    # 4. :SourceFile nodes (created by graph-rag enrichment)
    n_sf = _count(
        session,
        "MATCH (f:SourceFile) WHERE f.absolute_path ENDS WITH '.java' RETURN count(f) AS n",
    )
    if n_sf > 0:
        _ok(":SourceFile (.java) nodes present", f"{n_sf} files")
    else:
        _fail(
            ":SourceFile (.java) = 0  →  src/main/java not scanned or enrichment not run",
            "check .jqassistant.yml scan.directories and run main.py",
        )
        passed = False

    # 5. :WITH_SOURCE relationships (source-to-type links)
    n_ws = _count(session, "MATCH ()-[r:WITH_SOURCE]->() RETURN count(r) AS n")
    if n_ws > 0:
        _ok(":WITH_SOURCE relationships present", f"{n_ws}")
    else:
        _warn(":WITH_SOURCE = 0  →  enrichment (main.py) not yet run")

    # 6. Spring stereotype labels
    _header("🌱  Spring Stereotype Labels")
    for label, hint in [
        ("Controller", "@RestController / @Controller"),
        ("Service", "@Service"),
        ("Repository", "@Repository"),
    ]:
        n = _count(session, f"MATCH (c:{label}) RETURN count(c) AS n")
        if n > 0:
            _ok(f":{label} nodes present", f"{n}")
        else:
            _warn(f":{label} = 0  →  {hint} classes not found or concepts not applied")

    # 7. Layer labels (custom concepts)
    _header("🗂️   Layer Labels")
    for label in ("ApiLayer", "DomainLayer", "InfrastructureLayer"):
        n = _count(session, f"MATCH (c:{label}) RETURN count(c) AS n")
        if n > 0:
            _ok(f":{label} nodes present", f"{n}")
        else:
            _warn(f":{label} = 0")

    # 8. :INVOKES relationships
    n_inv = _count(session, "MATCH ()-[r:INVOKES]->() RETURN count(r) AS n")
    if n_inv > 0:
        _ok(":INVOKES relationships present", f"{n_inv}")
    else:
        _warn(":INVOKES = 0  →  call graph not available")

    # 9. Entity nodes (created by graph-rag enrichment)
    n_entity = _count(session, "MATCH (n:Entity) RETURN count(n) AS n")
    if n_entity > 0:
        _ok(":Entity nodes (graph-rag enrichment) present", f"{n_entity}")
    else:
        _warn(":Entity = 0  →  run main.py enrichment first")

    return passed


# ── Mode: config ──────────────────────────────────────────────────────────────


def check_config(session) -> bool:
    """Check graph configuration, APOC, label inventory, and enrichment state."""
    passed = True

    _header("⚙️   Configuration & Schema Check")

    # 1. APOC availability
    try:
        session.run("RETURN apoc.version() AS v").single()
        _ok("APOC plugin loaded")
    except Exception:
        _warn(
            "APOC plugin not available  →  some enrichment steps will fail",
            "add apoc-core to .jqassistant.yml neo4j-plugins",
        )

    # 2. All labels in the graph
    labels = [
        r["label"]
        for r in session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
    ]
    _info(f"Labels in graph ({len(labels)} total)", ", ".join(labels))

    required_labels = {"Java", "Type", "Class", "Method", "Package", "Artifact"}
    missing = required_labels - set(labels)
    if not missing:
        _ok("All required base labels present")
    else:
        _fail(f"Missing base labels: {missing}  →  run jqassistant scan+analyze")
        passed = False

    enriched_labels = {"SourceFile", "Entity"}
    missing_enriched = enriched_labels - set(labels)
    if not missing_enriched:
        _ok("Graph-rag enrichment labels present (SourceFile, Entity)")
    else:
        _warn(f"Enrichment labels missing: {missing_enriched}  →  run main.py")

    # 3. Relationship types
    rel_types = [
        r["relationshipType"]
        for r in session.run(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType"
        )
    ]
    _info(f"Relationship types ({len(rel_types)} total)", ", ".join(rel_types))

    required_rels = {"INVOKES", "DECLARES", "DEPENDS_ON"}
    missing_rels = required_rels - set(rel_types)
    if not missing_rels:
        _ok("Required relationship types present")
    else:
        _warn(f"Missing relationship types: {missing_rels}")

    enrichment_rels = {"WITH_SOURCE", "SIMILAR_TO"}
    missing_enrich_rels = enrichment_rels - set(rel_types)
    if not missing_enrich_rels:
        _ok("Enrichment relationship types present (WITH_SOURCE, SIMILAR_TO)")
    else:
        _warn(
            f"Enrichment relationships missing: {missing_enrich_rels}  →  run main.py"
        )

    # 4. Total node and relationship counts
    _header("📊  Node & Relationship Counts")
    total_nodes = _count(session, "MATCH (n) RETURN count(n) AS n")
    total_rels = _count(session, "MATCH ()-[r]->() RETURN count(r) AS n")
    _info(f"Total nodes", str(total_nodes))
    _info(f"Total relationships", str(total_rels))
    if total_nodes == 0:
        _fail("Graph is empty  →  run jqassistant scan first")
        passed = False
    elif total_nodes < 100:
        _warn(f"Only {total_nodes} nodes  →  scan may be incomplete")

    # 5. Embeddings
    _header("🔢  Embedding State")
    n_emb = _count(
        session, "MATCH (n) WHERE n.embedding IS NOT NULL RETURN count(n) AS n"
    )
    if n_emb > 0:
        _ok("Embedding vectors present", f"{n_emb} nodes")
    else:
        _warn(
            "No embeddings found  →  run: python3 main.py --generate-summary  (sets embeddings)"
        )

    # 6. Summaries
    n_sum = _count(
        session, "MATCH (n) WHERE n.summary IS NOT NULL RETURN count(n) AS n"
    )
    if n_sum > 0:
        _ok("Summary properties present", f"{n_sum} nodes")
    else:
        _warn(
            "No summaries found  →  run main.py --generate-summary with an LLM backend"
        )

    # 7. Constraint / index inventory
    _header("🔒  Constraints & Indexes")
    try:
        constraints = list(session.run("SHOW CONSTRAINTS YIELD name RETURN name"))
        indexes = list(session.run("SHOW INDEXES YIELD name RETURN name"))
        _info(f"Constraints defined", str(len(constraints)))
        _info(f"Indexes defined", str(len(indexes)))
    except Exception:
        _warn("Could not query constraints/indexes (requires Neo4j 4.4+)")

    return passed


# ── Entry point ───────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Graph health checker for jqassistant-graph-rag.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["java", "config"],
        default="java",
        help="Check mode: 'java' (analysis completeness) or 'config' (schema/configuration).",
    )
    parser.add_argument("--uri", default="bolt://localhost:7688")
    parser.add_argument("--user", default="")
    parser.add_argument("--password", default="")
    args = parser.parse_args()

    print(
        f"\n{BOLD}jqassistant-graph-rag  —  Graph Health Check  [{args.mode.upper()}]{RESET}"
    )
    print(f"Neo4j: {args.uri}\n")

    driver = connect(args.uri, args.user, args.password)
    with driver.session() as session:
        if args.mode == "java":
            ok = check_java(session)
        else:
            ok = check_config(session)
    driver.close()

    print()
    if ok:
        print(f"{GREEN}{BOLD}✅  All checks passed.{RESET}\n")
        sys.exit(0)
    else:
        print(
            f"{RED}{BOLD}❌  One or more checks failed — review the output above.{RESET}\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
