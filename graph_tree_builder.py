import logging
import os
from pathlib import Path
from collections import defaultdict
from neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class GraphTreeBuilder:
    """
    Handles the third phase of graph normalization: establishing a clean,
    hierarchical tree structure for the project.
    """

    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        self.project_path = None
        self.project_name = None
        logger.info("Initialized GraphTreeBuilder.")

    def create_project_node(self):
        """
        Auto-detects the project's root path, creates a single :Project node,
        and links all :Artifact nodes to it.
        """
        logger.info("--- Starting Pass: Create Project Node ---")

        # Prefer an explicit PROJECT_ROOT (set by the manager) if available.
        env_root = os.environ.get("PROJECT_ROOT") or os.environ.get("PROJECT_ROOT_PATH")
        if env_root:
            env_root_path = Path(env_root).expanduser().resolve()
            if env_root_path.exists():
                self.project_path = env_root_path
                self.project_name = self.project_path.name
                logger.info(f"Using PROJECT_ROOT from env: {self.project_path}")
            else:
                logger.warning(
                    f"PROJECT_ROOT env var set but path does not exist: {env_root}"
                )

        if not self.project_path:
            # Auto-detect project root from the graph. Get the top level directory nodes.
            # The top level directory nodes can be :Artifact or not, while mostly not.
            # Top level directory nodes should always have absolute path in fileName.
            query = """
            MATCH (d:Directory)
            WHERE NOT EXISTS { (parent_dir:Directory)-[:CONTAINS]->(d) }
            RETURN d.absolute_path AS path
            """
            results = self.neo4j_manager.execute_read_query(query)
            top_dir_paths = [res["path"] for res in results if res and res.get("path")]

            if top_dir_paths:
                project_path_str = os.path.commonpath(top_dir_paths)
                self.project_path = Path(project_path_str).resolve()
                self.project_name = self.project_path.name
                logger.info(f"Auto-detected project path: {self.project_path}")
            else:
                # Last-resort fallback to current working directory
                cwd = Path.cwd().resolve()
                self.project_path = cwd
                self.project_name = cwd.name
                logger.warning(
                    f"Could not detect project path from graph; falling back to CWD: {self.project_path}"
                )

        # 2. Create :Project node and link artifacts
        self.neo4j_manager.execute_write_query(
            """
            MERGE (p:Project {name: $projectName})
            ON CREATE SET p.creationTimestamp = datetime()
            SET p.absolute_path = $projectPath
            WITH p
            MATCH (a:Artifact) WHERE a:Directory|Jar
            MERGE (p)-[:CONTAINS]->(a)
            WITH p
            MATCH (d:Directory)
            WHERE NOT EXISTS { (parent_dir:Directory)-[:CONTAINS]->(d) }
            MERGE (p)-[:CONTAINS]->(d)
            """,
            params={
                "projectName": self.project_name,
                "projectPath": str(self.project_path),
            },
        )
        logger.info(
            f"Created :Project node for '{self.project_name}' and linked with artifacts and top-level directories."
        )
        logger.info("--- Finished Pass: Create Project Node ---")
        return self.project_path

    def _ensure_source_directories(self):
        """
        Creates :Directory nodes for every intermediate path segment between the
        project root and each :SourceFile node.

        jQAssistant creates package :Directory nodes relative to the compiled-class
        root (src/main/java), so paths like .../src/main/java/com/dkt/... do not
        exist as Directory nodes in the graph.  Without these intermediate nodes
        establish_source_hierarchy()'s STARTS_WITH depth-check never fires and no
        [:CONTAINS_SOURCE] relationships are created.
        """
        logger.info("--- Sub-pass: Ensure Source Directory Nodes ---")
        project_path_str = str(self.project_path)

        source_files = self.neo4j_manager.execute_read_query(
            """
            MATCH (sf:SourceFile)
            WHERE sf.absolute_path IS NOT NULL
              AND sf.absolute_path STARTS WITH $prefix
            RETURN sf.absolute_path AS path
            """,
            params={"prefix": project_path_str + "/"},
        )

        if not source_files:
            logger.warning(
                "No SourceFile nodes found under project path — skipping directory creation."
            )
            return

        dir_paths: set[str] = set()
        for record in source_files:
            p = Path(record["path"]).parent  # start at the file's immediate parent
            while str(p).startswith(project_path_str) and p != self.project_path:
                dir_paths.add(str(p))
                p = p.parent

        if not dir_paths:
            logger.info("All intermediate source directories already exist.")
            return

        self.neo4j_manager.execute_write_query(
            """
            UNWIND $paths AS dir_path
            MERGE (d:Directory {absolute_path: dir_path})
            ON CREATE SET d.fileName = dir_path
            """,
            params={"paths": list(dir_paths)},
        )
        logger.info(
            f"Ensured {len(dir_paths)} intermediate source :Directory nodes exist."
        )

    def establish_source_hierarchy(self):
        """
        Establishes a direct hierarchical structure for source entities
        using [:CONTAINS_SOURCE], processing level by level.
        """
        if not self.project_path:
            raise ValueError(
                "Project path has not been determined. Run create_project_node() first."
            )

        logger.info("--- Starting Pass: Establish Direct Source Hierarchy ---")

        # Guarantee that every path segment between the project root and each
        # SourceFile has a corresponding :Directory node, so the STARTS WITH
        # depth-checks below can match them.
        self._ensure_source_directories()

        # 1. Get all directories with their depths
        query_all_dirs = """
        MATCH (d:Directory)
        WHERE d.absolute_path IS NOT NULL
        RETURN d.absolute_path AS path, size(split(d.absolute_path, '/')) AS depth
        """
        all_dirs_with_depth = self.neo4j_manager.execute_read_query(query_all_dirs)

        if not all_dirs_with_depth:
            logger.warning(
                "No directories with absolute_path found to establish hierarchy."
            )
            return

        # Link directories to their direct SourceFile children
        self.neo4j_manager.execute_write_query(
            """
            UNWIND $paths AS dir_path
            MATCH (parentDir:Directory {absolute_path: dir_path})
            MATCH (sf:SourceFile)
            WHERE sf.absolute_path STARTS WITH parentDir.absolute_path + '/'
                AND size(split(sf.absolute_path, '/')) = size(split(parentDir.absolute_path, '/')) + 1
            MERGE (parentDir)-[:CONTAINS_SOURCE]->(sf)
            """,
            params={"paths": [d["path"] for d in all_dirs_with_depth]},
        )

        dirs_by_depth = defaultdict(list)
        for item in all_dirs_with_depth:
            dirs_by_depth[item["depth"]].append(item["path"])

        # 2. Process levels from deepest to shallowest
        for depth in sorted(dirs_by_depth.keys(), reverse=True):
            current_depth_dir_paths = dirs_by_depth[depth]

            # Link directories to their direct Directory children
            self.neo4j_manager.execute_write_query(
                """
                UNWIND $paths AS parent_path
                MATCH (parentDir:Directory {absolute_path: parent_path})
                MATCH (childDir:Directory)
                WHERE childDir.absolute_path STARTS WITH parentDir.absolute_path + '/'
                  AND size(split(childDir.absolute_path, '/')) = size(split(parentDir.absolute_path, '/')) + 1
                  AND EXISTS {(childDir)-[:CONTAINS_SOURCE]->()}
                MERGE (parentDir)-[:CONTAINS_SOURCE]->(childDir)
                """,
                params={"paths": current_depth_dir_paths},
            )

        logger.info(
            "Established [:CONTAINS_SOURCE] relationships between directories and source files."
        )

        # 3. Link Project node to top-level directories
        self.neo4j_manager.execute_write_query(
            """
            MATCH (p:Project {absolute_path: $projectPath})
            MATCH (d:Directory)
            WHERE EXISTS {(d)-[:CONTAINS_SOURCE]->()}
            AND NOT EXISTS {(parent_dir:Directory)-[:CONTAINS]->(d)}
            MERGE (p)-[:CONTAINS_SOURCE]->(d)
            """,
            params={"projectPath": str(self.project_path)},
        )
        logger.info("Linked :Project node to top-level source directories.")
        logger.info("--- Finished Pass: Establish Direct Source Hierarchy ---")
