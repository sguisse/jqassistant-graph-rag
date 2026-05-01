import logging
from neo4j_manager import Neo4jManager
from graph_basic_normalizer import GraphBasicNormalizer
from source_file_linker import SourceFileLinker
from graph_tree_builder import GraphTreeBuilder
from graph_entity_setter import GraphEntitySetter
from artifact_data_normalizer import ArtifactDataNormalizer

logger = logging.getLogger(__name__)


class GraphOrchestrator:
    """
    Manages and executes the sequence of graph normalization and enrichment passes.
    """

    def __init__(self, neo4j_manager: Neo4jManager, repo_root: str = ""):
        self.neo4j_manager = neo4j_manager
        self.project_path = None
        self.repo_root = repo_root or ""
        logger.info("Initialized GraphOrchestrator.")

    def run_enrichment_passes(self):
        """
        Executes the full sequence of graph enrichment passes by instantiating
        and running the necessary components in the correct logical order.
        """
        logger.info("--- Starting All Graph Enrichment and Normalization Passes ---")

        # --- Pre-flight: check the graph has been scanned ---
        # If jqassistant:scan has not been run, no :Type nodes exist.
        # Without types, all source-linking and summarization passes produce
        # nothing useful and emit dozens of confusing "unknown label" warnings.
        type_check = self.neo4j_manager.execute_read_query(
            "MATCH (t:Java:Type) RETURN count(t) AS n LIMIT 1"
        )
        type_count = type_check[0]["n"] if type_check else 0
        if type_count == 0:
            logger.warning(
                "PRE-FLIGHT CHECK FAILED: No Java type nodes (:Java:Type) found in the graph. "
                "Source-file linking and summarization passes will produce 0 results. "
                "Run 'B4 — jqassistant scan + analyze' from the manager before running enrichment."
            )

        # Instantiate all the specialized handlers
        basic_normalizer = GraphBasicNormalizer(self.neo4j_manager)
        source_linker = SourceFileLinker(self.neo4j_manager)
        tree_builder = GraphTreeBuilder(self.neo4j_manager)
        artifact_normalizer = ArtifactDataNormalizer(self.neo4j_manager)
        entity_setter = GraphEntitySetter(self.neo4j_manager)

        # --- Phase 1: Basic Normalization ---
        # Add absolute paths and label source files first. This is a prerequisite
        # for almost all subsequent passes.
        basic_normalizer.add_absolute_paths()
        basic_normalizer.label_source_files()

        # --- Phase 2: Source Code Integration ---
        # With source files clearly labeled, we can now link types and members
        # to their on-disk source code.
        source_linker.link_types_to_source_files()
        source_linker.link_members_to_source_files()

        # --- Phase 3: Hierarchical Structure Establishment ---
        # Now that the graph is normalized and linked, build the clean
        # hierarchical overlay for the project.
        self.project_path = tree_builder.create_project_node(repo_root=self.repo_root)
        tree_builder.establish_source_hierarchy()

        # --- Phase 4: Artifact & Package Data Normalization ---
        # Correct the core artifact structure and build the class hierarchy overlay.
        artifact_normalizer.merge_duplicate_types()
        artifact_normalizer.relocate_directory_artifacts()
        artifact_normalizer.rewrite_containment_relationships()
        artifact_normalizer.rewrite_requirement_relationships()
        artifact_normalizer.establish_class_hierarchy()
        artifact_normalizer.cleanup_package_semantics()
        artifact_normalizer.link_project_to_artifacts()

        # --- Phase 5: Entity and ID Generation ---
        # As the final step, label all relevant nodes as :Entity and generate
        # their stable, unique IDs for future processing.
        entity_setter.create_entities_and_stable_ids()

        logger.info("--- All Graph Enrichment and Normalization Passes Complete ---")
