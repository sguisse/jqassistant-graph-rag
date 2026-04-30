import logging
import os
from pathlib import Path
from neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class GraphBasicNormalizer:
    """
    Handles the first phase of graph normalization: adding canonical path
    information and identifying source files.
    """

    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        logger.info("Initialized GraphBasicNormalizer.")

    def add_absolute_paths(self):
        """
        Adds an 'absolute_path' property to all filesystem nodes, including
        the top-level artifacts themselves.
        """
        logger.info("--- Starting Pass: Add Absolute Paths ---")

        # Set absolute_path for top-level :Directory nodes AND all their descendants.
        #
        # jQAssistant creates a HIERARCHICAL :CONTAINS tree, not a flat one:
        #   src/main/java  -[:CONTAINS]->  com/  -[:CONTAINS]->  dkt/  -[:CONTAINS]->  Foo.java
        # A one-hop MATCH would miss files nested more than one level deep.
        #
        # jQAssistant stores each node's fileName as the path RELATIVE TO THE SCAN ROOT,
        # so for Foo.java under a scan root of src/main/java:
        #   root.fileName  = /absolute/path/to/src/main/java
        #   Foo.java.fileName = /com/dkt/smartassessment/Foo.java
        # => absolute_path = root.fileName + Foo.java.fileName  (correct full path)
        #
        # Step 1a: set absolute_path on root Directory nodes.
        root_query = """
        MATCH (e:Directory)
        WHERE NOT EXISTS { (:Directory)-[:CONTAINS]->(e) }
          AND e.fileName IS NOT NULL
        SET e.absolute_path = e.fileName
        RETURN count(e) AS roots_set
        """
        root_result = self.neo4j_manager.execute_write_query(root_query)
        logger.info(
            f"Set absolute_path on {root_result.properties_set} root Directory nodes."
        )

        # Step 1b: use [:CONTAINS*] to reach ALL descendants (any depth) and set their
        # absolute_path = root.absolute_path + descendant.fileName.
        directory_tree_query = """
        MATCH (e:Directory)
        WHERE NOT EXISTS { (:Directory)-[:CONTAINS]->(e) }
          AND e.absolute_path IS NOT NULL
        MATCH (e)-[:CONTAINS*]->(n)
        WHERE n.fileName IS NOT NULL
        SET n.absolute_path = e.absolute_path + n.fileName
        RETURN count(n) AS paths_normalized
        """

        dir_tree_result = self.neo4j_manager.execute_write_query(directory_tree_query)
        dir_tree_props_set = dir_tree_result.properties_set
        logger.info(
            f"Set 'absolute_path' for {dir_tree_props_set} descendant nodes (all depths)."
        )

        # Continue with artifact/contained/non-artifact normalization; do not return early

        artifact_query = """
        MATCH (e:Artifact&Directory)
        WHERE e.fileName IS NOT NULL
        SET e.absolute_path = e.fileName
        RETURN count(e) AS paths_normalized
        """
        artifact_result = self.neo4j_manager.execute_write_query(artifact_query)
        artifact_props_set = artifact_result.properties_set
        logger.info(
            f"Set 'absolute_path' for {artifact_props_set} Artifact:Directory nodes."
        )

        # 1.2. Second, set the path for the files and directories contained within them.
        # Since all the descendant nodes are contained directly by the Artifact:Directory nodes,
        # and the containership does not cross Artifact boundaries (to the nodes under an :Artifact:Jar node),
        # we can simply concatenate the parent's absolute_path with the children nodes' fileName.
        contained_query = """
        MATCH (e:Artifact&Directory)-[:CONTAINS]->(f:File|Directory)
        WHERE e.absolute_path IS NOT NULL AND f.fileName IS NOT NULL
        SET f.absolute_path = CASE
            WHEN f.fileName STARTS WITH '/' THEN e.absolute_path + f.fileName
            ELSE e.absolute_path + '/' + f.fileName
        END
        RETURN count(f) AS paths_normalized
        """
        contained_result = self.neo4j_manager.execute_write_query(contained_query)
        contained_props_set = contained_result.properties_set
        logger.info(
            f"Set 'absolute_path' for {contained_props_set} contained File/Directory nodes."
        )

        logger.info("--- Finished Pass: Add Absolute Paths ---")

        # If jQAssistant did not label any :SourceFile nodes (e.g. the graph only
        # contains compiled bytecode), fall back to scanning the workspace on disk.
        # Important: resolve the project root from PROJECT_ROOT env var (set by the
        # manager) or by walking up from cwd looking for pom.xml / .jqassistant.yml.
        try:
            root = self._find_project_root()
            if root is None:
                logger.warning(
                    "Disk-scan fallback skipped: could not determine project root "
                    "(set PROJECT_ROOT env var or run from the project directory)."
                )
                return
            candidate_dirs = [
                os.path.join(root, "src", "main", "java"),
                os.path.join(root, "src", "test", "java"),
                os.path.join(root, "src", "main", "kotlin"),
                os.path.join(root, "src"),
            ]

            found_files = []
            for d in candidate_dirs:
                if not os.path.isdir(d):
                    continue
                for dirpath, _, filenames in os.walk(d):
                    for fn in filenames:
                        if fn.endswith(".java") or fn.endswith(".kt"):
                            p = os.path.join(dirpath, fn)
                            found_files.append({"path": p, "name": fn})

            if found_files:
                logger.info(
                    f"Found {len(found_files)} source files on disk; registering in graph."
                )
                cypher = """
                UNWIND $files AS x
                MERGE (f:File {absolute_path: x.path})
                SET f:SourceFile, f.fileName = x.name
                """
                counters = self.neo4j_manager.execute_write_query(
                    cypher, params={"files": found_files}
                )
                logger.info(
                    f"Disk-sourced :SourceFile nodes created: {getattr(counters, 'nodes_created', 0)}"
                )
            else:
                logger.info("No source files found on disk during fallback scan.")
        except Exception as e:
            logger.warning(f"Fallback disk scan for source files failed: {e}")

    @staticmethod
    def _find_project_root() -> str | None:
        """
        Resolve the Maven project root directory.
        Priority:
          1. PROJECT_ROOT environment variable (set by jqassistant_manager.py)
          2. Walk up from cwd looking for pom.xml or .jqassistant.yml
        """
        env_root = os.environ.get("PROJECT_ROOT", "").strip()
        if env_root and os.path.isdir(env_root):
            return env_root
        current = os.path.abspath(os.getcwd())
        for _ in range(12):
            for marker in ("pom.xml", ".jqassistant.yml", ".git"):
                if os.path.exists(os.path.join(current, marker)):
                    return current
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return None

    def label_source_files(self):
        """
        Identifies and labels :File nodes that represent Java or Kotlin
        source code files as :SourceFile.
        This pass relies on 'absolute_path' having been set previously.
        """
        logger.info("--- Starting Pass: Label Source Files ---")
        # Match case-insensitively and tolerate leading/trailing whitespace
        query = """
        MATCH (f:File)
        WHERE f.absolute_path IS NOT NULL
        AND (toLower(trim(f.absolute_path)) ENDS WITH '.java' OR toLower(trim(f.absolute_path)) ENDS WITH '.kt')
        SET f:SourceFile
        RETURN count(f) AS source_files_labeled
        """
        result = self.neo4j_manager.execute_write_query(query)
        labels_added = result.labels_added
        logger.info(f"Labeled {labels_added} files as :SourceFile.")
        logger.info("--- Finished Pass: Label Source Files ---")
