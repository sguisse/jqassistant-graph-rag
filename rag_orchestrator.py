import logging
import os
import re
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from neo4j_manager import Neo4jManager
from method_analyzer import MethodAnalyzer
from method_summarizer import MethodSummarizer
from type_summarizer import TypeSummarizer
from source_file_summarizer import SourceFileSummarizer
from directory_summarizer import DirectorySummarizer
from package_summarizer import PackageSummarizer
from project_summarizer import ProjectSummarizer
from entity_embedder import EntityEmbedder
from llm_client import (
    get_llm_client,
    get_embedding_client,
    LlmClient,
    EmbeddingClient,
    FakeLlmClient,
    set_global_llm_total,
    set_pass_total,
    reset_global_llm_progress,
    reset_label_call_counts,
    reset_all_pass_progress,
)
from summary_cache_manager import SummaryCacheManager
from node_summary_processor import NodeSummaryProcessor
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class _CounterPlan:
    global_total: int = 0
    per_label_totals: dict[str, int] = field(default_factory=dict)


class _PlanningLlmClient(LlmClient):
    """Silent fake LLM used to simulate the run and count exact LLM calls."""

    def __init__(self):
        self._fake = FakeLlmClient()
        self.global_total = 0
        self.per_label_totals: dict[str, int] = defaultdict(int)
        self._label_call_counts: dict[str, int] = defaultdict(int)

    @staticmethod
    def _camel_to_upper_snake(name: str) -> str:
        s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
        s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
        return s2.upper()

    def _get_label_max(self, label: str) -> int | None:
        snake = self._camel_to_upper_snake(label)
        raw = os.environ.get(f"{snake}_MAX_LLM_CALL") or os.environ.get("MAX_LLM_CALL")
        if raw:
            try:
                value = int(raw)
                return value if value > 0 else None
            except ValueError:
                return None
        return None

    def generate_summary(
        self, prompt: str, index: int = 1, total: int = 1, label: str = ""
    ) -> str:
        if label:
            max_calls = self._get_label_max(label)
            current = self._label_call_counts[label]
            if max_calls is not None and current >= max_calls:
                return ""
            self._label_call_counts[label] = current + 1
            self.per_label_totals[label] += 1

        self.global_total += 1
        return self._fake._call_llm(prompt, index, total)

    def _call_llm(self, prompt: str, index: int = 1, total: int = 1) -> str:
        return self._fake._call_llm(prompt, index, total)


class RagOrchestrator:
    """
    Manages and executes the sequence of RAG (summary and embedding) generation passes.
    """

    def __init__(self, neo4j_manager: Neo4jManager, project_path: Path, llm_api: str):
        self.neo4j_manager = neo4j_manager
        self.project_path = project_path
        self.project_name = self.project_path.name
        self.llm_api = llm_api

        # Initialize core components
        self.llm_client: LlmClient = get_llm_client(self.llm_api)
        self.embedding_client: EmbeddingClient = get_embedding_client(
            "sentence-transformer"
        )
        self.cache_manager = SummaryCacheManager(str(self.project_path))
        self.node_summary_processor = NodeSummaryProcessor(
            self.llm_client, self.cache_manager
        )

        # Initialize all pass handlers with the necessary components
        self.method_analyzer = MethodAnalyzer(
            neo4j_manager, self.node_summary_processor
        )
        self.method_summarizer = MethodSummarizer(
            neo4j_manager, self.node_summary_processor
        )
        self.type_summarizer = TypeSummarizer(
            neo4j_manager, self.node_summary_processor
        )
        self.source_file_summarizer = SourceFileSummarizer(
            neo4j_manager, self.node_summary_processor
        )
        self.directory_summarizer = DirectorySummarizer(
            neo4j_manager, self.node_summary_processor
        )
        self.package_summarizer = PackageSummarizer(
            neo4j_manager, self.node_summary_processor
        )
        self.project_summarizer = ProjectSummarizer(
            neo4j_manager, self.node_summary_processor
        )
        self.entity_embedder = EntityEmbedder(neo4j_manager, self.embedding_client)

        logger.info(
            f"Initialized RagOrchestrator for project: {self.project_name} with LLM API: {self.llm_api}"
        )

    def run_rag_passes(self):
        """
        Executes the full sequence of RAG generation passes with caching.
        """
        # Pre-flight: skip all passes if the graph has no Java type nodes.
        # Without a prior jqassistant:scan, labels like :Method, :Class and
        # relationships like :WITH_SOURCE and :INVOKES do not exist in the schema.
        # Running any RAG pass would produce dozens of confusing "unknown label /
        # relationship / property" DBMS notifications and zero useful output.
        type_check = self.neo4j_manager.execute_read_query(
            "MATCH (t:Java:Type) RETURN count(t) AS n LIMIT 1"
        )
        type_count = type_check[0]["n"] if type_check else 0
        if type_count == 0:
            logger.warning(
                "RAG SKIPPED: No Java type nodes (:Java:Type) found in the graph. "
                "Summarization passes require a prior jqassistant scan to produce results. "
                "Run 'B4 — jqassistant scan + analyze' from the manager first."
            )
            return

        self.cache_manager.load()
        progress_plan: Optional[_CounterPlan] = None
        # Build an exact progress plan by simulating the run with a silent fake
        # LLM and a cloned in-memory cache state.
        try:
            reset_global_llm_progress()
            reset_label_call_counts()
            reset_all_pass_progress()
            progress_plan = self._build_exact_progress_plan()
            if progress_plan.global_total > 0:
                set_global_llm_total(progress_plan.global_total)
                logger.info(
                    "Exact LLM progress plan built: %d calls across %d passes.",
                    progress_plan.global_total,
                    len(progress_plan.per_label_totals),
                )
        except Exception:
            logger.exception(
                "Failed to build exact progress plan; continuing without a global denominator."
            )
        try:
            logger.info(
                f"--- Starting All RAG Generation Passes for project: {self.project_name} ---"
            )

            # The sequence of passes remains the same
            self._initialize_pass_progress(progress_plan, "MethodAnalyzer")
            self.method_analyzer.run()
            self._initialize_pass_progress(progress_plan, "MethodSummarizer")
            self.method_summarizer.run()
            self._initialize_pass_progress(progress_plan, "TypeSummarizer")
            self.type_summarizer.run()
            self._initialize_pass_progress(progress_plan, "SourceFileSummarizer")
            self.source_file_summarizer.run()
            self._initialize_pass_progress(progress_plan, "DirectorySummarizer")
            self.directory_summarizer.run()
            self._initialize_pass_progress(progress_plan, "PackageSummarizer")
            self.package_summarizer.run()
            self._initialize_pass_progress(progress_plan, "ProjectSummarizer")
            self.project_summarizer.run()
            self.entity_embedder.add_entity_labels_and_embeddings()

            logger.info(
                f"--- All RAG Generation Passes for project: {self.project_name} Complete ---"
            )
        finally:
            # Ensure the cache is saved even if an error occurs
            self.cache_manager.save()

    @staticmethod
    def _initialize_pass_progress(
        progress_plan: Optional[_CounterPlan], label: str
    ) -> None:
        if not progress_plan:
            return
        total = progress_plan.per_label_totals.get(label)
        if total is None:
            return
        set_pass_total(label, total)

    @staticmethod
    def _apply_simulated_result(
        planning_cache: SummaryCacheManager, result: Optional[dict[str, Any]]
    ) -> None:
        if not result:
            return

        node_id = result.get("id")
        if not node_id:
            return

        cache_data = {}
        if "summary" in result:
            cache_data["summary"] = result["summary"]
        if "code_analysis" in result:
            cache_data["code_analysis"] = result["code_analysis"]
        if "code_hash" in result:
            cache_data["code_hash"] = result["code_hash"]

        if cache_data:
            planning_cache.update_node_cache(node_id, cache_data)

        if result.get("status") == "regenerated":
            planning_cache.set_runtime_status(node_id, "regenerated")

    def _simulate_batch(
        self,
        items: list[dict[str, Any]],
        processor_fn: Callable[[dict[str, Any]], Optional[dict[str, Any]]],
        planning_cache: SummaryCacheManager,
        prepare_fn: Optional[Callable[[dict[str, Any]], dict[str, Any]]] = None,
    ) -> None:
        raw_results: list[Optional[dict[str, Any]]] = []
        for item in items:
            prepared = dict(item)
            if prepare_fn:
                prepared = prepare_fn(prepared)
            raw_results.append(processor_fn(prepared))

        for raw_result in raw_results:
            self._apply_simulated_result(planning_cache, raw_result)

    def _build_exact_progress_plan(self) -> _CounterPlan:
        planning_cache = SummaryCacheManager(str(self.project_path))
        planning_cache.cache = deepcopy(self.cache_manager.cache)
        planning_cache.runtime_status = deepcopy(self.cache_manager.runtime_status)

        planning_llm = _PlanningLlmClient()
        planning_processor = NodeSummaryProcessor(planning_llm, planning_cache)

        # Pass 1: MethodAnalyzer
        analyzer_items = self.neo4j_manager.execute_read_query(
            self.method_analyzer._get_items_query(),
            params={"analysisProperty": "code_analysis", "hashProperty": "code_hash"},
        )
        self._simulate_batch(
            analyzer_items,
            planning_processor.get_method_code_analysis,
            planning_cache,
            prepare_fn=self.method_analyzer._prepare_item,
        )

        # Pass 2: MethodSummarizer — planning query intentionally does not rely
        # on db code_analysis because the simulation cache now contains the
        # results of the planned analyzer pass.
        method_items = self.neo4j_manager.execute_read_query(
            """
            MATCH (m:Method)
            WHERE m.entity_id IS NOT NULL
            OPTIONAL MATCH (caller:Method)-[:INVOKES]->(m)
            OPTIONAL MATCH (m)-[:INVOKES]->(callee:Method)
            RETURN m.entity_id AS id,
                   m.name AS name,
                   m.summary AS db_summary,
                   collect(DISTINCT caller.entity_id) AS callers,
                   collect(DISTINCT callee.entity_id) AS callees
            """
        )
        self._simulate_batch(
            method_items,
            planning_processor.get_method_summary,
            planning_cache,
        )

        # Pass 3: TypeSummarizer — process inheritance levels exactly like the real pass.
        types_by_level = self.type_summarizer._get_types_by_inheritance_level()
        for level in sorted(types_by_level.keys()):
            items_to_process = self.type_summarizer._get_context_for_ids(
                types_by_level[level]
            )
            self._simulate_batch(
                items_to_process,
                planning_processor.get_type_summary,
                planning_cache,
                prepare_fn=self.type_summarizer._prepare_item,
            )

        # Pass 4: SourceFileSummarizer — planning query collects all linked types;
        # the simulated cache decides whether child summaries are available.
        source_file_items = self.neo4j_manager.execute_read_query(
            """
            MATCH (sf:SourceFile)
            OPTIONAL MATCH (sf)<-[:WITH_SOURCE]-(t:Type)
            RETURN sf.entity_id AS id,
                   sf.absolute_path AS path,
                   sf.summary AS db_summary,
                   collect(DISTINCT t.entity_id) AS dependency_ids
            """
        )
        self._simulate_batch(
            source_file_items,
            lambda item: planning_processor.get_hierarchical_summary(
                item, "SourceFile"
            ),
            planning_cache,
        )

        # Pass 5: DirectorySummarizer — preserve real bottom-up depth ordering.
        directories_by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in self.directory_summarizer._get_directories_ordered_by_depth():
            directories_by_depth[item["depth"]].append(item)
        for depth in sorted(directories_by_depth.keys(), reverse=True):
            self._simulate_batch(
                directories_by_depth[depth],
                lambda item: planning_processor.get_hierarchical_summary(
                    item, "Directory"
                ),
                planning_cache,
            )

        # Pass 6a: PackageSummarizer internal packages.
        package_items = self.neo4j_manager.execute_read_query(
            """
            MATCH (a:Artifact)-[:CONTAINS_CLASS*]->(p:Package)
            WHERE p.fqn IS NOT NULL AND p.summary IS NULL
            WITH p, size(split(p.fqn, '.')) AS depth
            OPTIONAL MATCH (p)-[:CONTAINS_CLASS]->(child)
            WHERE child:Package OR child:Type
            RETURN
                p.entity_id AS id,
                p.fqn AS fqn,
                p.summary AS db_summary,
                collect(DISTINCT child.entity_id) AS dependency_ids,
                depth
            ORDER BY depth DESC
            """
        )
        packages_by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in package_items:
            packages_by_depth[item["depth"]].append(item)
        for depth in sorted(packages_by_depth.keys(), reverse=True):
            self._simulate_batch(
                packages_by_depth[depth],
                lambda item: planning_processor.get_hierarchical_summary(
                    item, "Package"
                ),
                planning_cache,
            )

        # Pass 6b: PackageSummarizer artifact roots.
        artifact_items = self.neo4j_manager.execute_read_query(
            """
            MATCH (a:Artifact)
            WHERE a.summary IS NULL
            OPTIONAL MATCH (a)-[:CONTAINS_CLASS]->(child)
            WHERE child:Package OR child:Type
            RETURN
                a.entity_id AS id,
                a.fileName AS path,
                a.summary AS db_summary,
                collect(DISTINCT child.entity_id) AS dependency_ids
            """
        )
        self._simulate_batch(
            artifact_items,
            lambda item: planning_processor.get_hierarchical_summary(item, "Package"),
            planning_cache,
        )

        # Pass 7: ProjectSummarizer.
        self._simulate_batch(
            self.project_summarizer._get_project_with_context(),
            planning_processor.get_project_summary,
            planning_cache,
        )

        return _CounterPlan(
            global_total=planning_llm.global_total,
            per_label_totals=dict(planning_llm.per_label_totals),
        )

    def _compute_estimated_prompt_total(self) -> int:
        """Estimate the total number of LLM prompts that will be issued during
        the run by querying Neo4j for counts of the main node types that are
        summarized. This is an approximation (some nodes may produce multiple
        iterative prompts), but it gives a useful global total for progress.
        """
        total = 0

        def _count(query: str, params: dict | None = None) -> int:
            try:
                res = self.neo4j_manager.execute_read_query(query, params=params or {})
                if res and isinstance(res, list) and len(res) > 0:
                    return int(res[0].get("n", 0))
            except Exception:
                logger.debug("Count query failed", exc_info=True)
            return 0

        # ── Pass 1: MethodAnalyzer (code_analysis) ───────────────────────────
        # Mirrors MethodAnalyzer._get_items_query(): methods with source lines.
        # Only methods with code_analysis IS NULL need LLM regeneration.
        total += _count(
            """MATCH (m:Method)-[:WITH_SOURCE]->(:SourceFile)
               WHERE m.entity_id IS NOT NULL
                 AND m.firstLineNumber IS NOT NULL
                 AND m.lastLineNumber IS NOT NULL
                 AND m.code_analysis IS NULL
               RETURN count(m) AS n"""
        )

        # ── Pass 2: MethodSummarizer ──────────────────────────────────────────
        # Mirrors MethodSummarizer._get_items_query(): all methods with entity_id.
        # Only methods with summary IS NULL need LLM regeneration.
        total += _count(
            """MATCH (m:Method)
               WHERE m.entity_id IS NOT NULL
                 AND m.summary IS NULL
               RETURN count(m) AS n"""
        )

        # ── Pass 3: TypeSummarizer ────────────────────────────────────────────
        # Mirrors TypeSummarizer._get_source_linked_items_query().
        total += _count(
            """MATCH (t:Type)-[:WITH_SOURCE]->(:SourceFile)
               WHERE t.summary IS NULL
               RETURN count(DISTINCT t) AS n"""
        )

        # ── Pass 4: SourceFileSummarizer ──────────────────────────────────────
        # Mirrors SourceFileSummarizer._get_items_query().
        total += _count(
            "MATCH (sf:SourceFile) WHERE sf.summary IS NULL RETURN count(sf) AS n"
        )

        # ── Pass 5: DirectorySummarizer ───────────────────────────────────────
        # Mirrors DirectorySummarizer._get_all_directories_query().
        total += _count(
            """MATCH (d:Directory)
               WHERE d.absolute_path IS NOT NULL
                 AND d.entity_id IS NOT NULL
                 AND d.summary IS NULL
               RETURN count(d) AS n"""
        )

        # ── Pass 6a: PackageSummarizer (packages) ──────────────────────────────
        # The real query already filters p.summary IS NULL.
        total += _count(
            """MATCH (a:Artifact)-[:CONTAINS_CLASS*]->(p:Package)
               WHERE p.fqn IS NOT NULL
                 AND p.summary IS NULL
               RETURN count(DISTINCT p) AS n"""
        )

        # ── Pass 6b: PackageSummarizer (artifacts) ────────────────────────────
        # The real query already filters a.summary IS NULL.
        total += _count(
            "MATCH (a:Artifact) WHERE a.summary IS NULL RETURN count(a) AS n"
        )

        # ── Pass 7: ProjectSummarizer ─────────────────────────────────────────
        total += _count(
            """MATCH (p:Project)
               WHERE p.entity_id IS NOT NULL
                 AND p.summary IS NULL
               RETURN count(p) AS n"""
        )

        return total
