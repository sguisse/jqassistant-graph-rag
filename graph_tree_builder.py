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

    def create_project_node(self, repo_root: str = ""):
        """
        Auto-detects the project's root path, creates a single :Project node,
        and links all :Artifact nodes to it.

        Args:
            repo_root: Explicit path to the repository root (preferred). When
                       provided the expensive commonpath query is skipped and
                       the project is named after the directory instead of the
                       common ancestor of all top-level nodes (which can be the
                       home directory when multiple artifact trees are present).
        """
        logger.info("--- Starting Pass: Create Project Node ---")

        if repo_root:
            # Use the explicitly provided root — most reliable
            self.project_path = Path(repo_root).resolve()
        else:
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
            else:
                self.project_path = Path.cwd().resolve()

        # Also check PROJECT_ROOT env var as fallback when no arg given
        if not repo_root:
            env_root = os.environ.get("PROJECT_ROOT", "")
            if env_root and Path(env_root).is_dir():
                self.project_path = Path(env_root).resolve()

        self.project_name = self.project_path.name
        logger.info(f"Auto-detected project path: {self.project_path}")

        # 2a. Remove stale Project nodes from previous runs that used a different
        #     project name (e.g., 'mac-SGUISS21' from the old commonpath fallback).
        #     Keeping them causes entity_id uniqueness conflicts in the entity setter.
        self.neo4j_manager.execute_write_query(
            "MATCH (p:Project) WHERE p.name <> $projectName DETACH DELETE p",
            params={"projectName": self.project_name},
        )

        # 2b. Create (or merge) the canonical :Project node and link artifacts
        self.neo4j_manager.execute_write_query(
            """
            MERGE (p:Project {name: $projectName})
            ON CREATE SET p.creationTimestamp = datetime()
            SET p.absolute_path = $projectPath
            WITH p
            MATCH (a:Artifact) WHERE 'Jar' IN labels(a) OR a:Directory
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

        self._ensure_source_directories_exist()

        query_all_dirs = """
        MATCH (d:Directory)
        WHERE d.absolute_path IS NOT NULL
          AND d.absolute_path STARTS WITH $projectPath
        RETURN d.absolute_path AS path, size(split(d.absolute_path, '/')) AS depth
        """
        all_dirs_with_depth = self.neo4j_manager.execute_read_query(
            query_all_dirs,
            params={"projectPath": str(self.project_path)},
        )

        if not all_dirs_with_depth:
            logger.warning(
                "No directories with absolute_path found to establish hierarchy."
            )
            return

        self.neo4j_manager.execute_write_query(
            """
            UNWIND $paths AS dir_path
            MATCH (parentDir:Directory {absolute_path: dir_path})
            WITH parentDir,
                 CASE WHEN dir_path ENDS WITH '/'
                      THEN left(dir_path, size(dir_path) - 1)
                      ELSE dir_path END AS normDir
            MATCH (sf:SourceFile)
            WHERE sf.absolute_path STARTS WITH $projectPath
              AND sf.absolute_path STARTS WITH normDir + '/'
              AND size(split(sf.absolute_path, '/')) = size(split(normDir, '/')) + 1
            MERGE (parentDir)-[:CONTAINS_SOURCE]->(sf)
            """,
            params={
                "paths": [item["path"] for item in all_dirs_with_depth],
                "projectPath": str(self.project_path),
            },
        )

        dirs_by_depth = defaultdict(list)
        for item in all_dirs_with_depth:
            dirs_by_depth[item["depth"]].append(item["path"])

        for depth in sorted(dirs_by_depth.keys(), reverse=True):
            current_depth_dir_paths = dirs_by_depth[depth]
            self.neo4j_manager.execute_write_query(
                """
                UNWIND $paths AS parent_path
                MATCH (parentDir:Directory {absolute_path: parent_path})
                WITH parentDir,
                     CASE WHEN parent_path ENDS WITH '/'
                          THEN left(parent_path, size(parent_path) - 1)
                          ELSE parent_path END AS normParent
                MATCH (childDir:Directory)
                WHERE childDir.absolute_path STARTS WITH $projectPath
                  AND childDir.absolute_path STARTS WITH normParent + '/'
                  AND size(split(childDir.absolute_path, '/')) = size(split(normParent, '/')) + 1
                  AND EXISTS {(childDir)-[:CONTAINS_SOURCE]->()}
                MERGE (parentDir)-[:CONTAINS_SOURCE]->(childDir)
                """,
                params={
                    "paths": current_depth_dir_paths,
                    "projectPath": str(self.project_path),
                },
            )

        logger.info(
            "Established [:CONTAINS_SOURCE] relationships between directories and source files."
        )

        self.neo4j_manager.execute_write_query(
            """
            MATCH (p:Project {absolute_path: $projectPath})
            MATCH (d:Directory)
            WHERE d.absolute_path STARTS WITH $projectPath
              AND EXISTS {(d)-[:CONTAINS_SOURCE]->()}
              AND NOT EXISTS {(parent_dir:Directory)-[:CONTAINS_SOURCE]->(d)}
            MERGE (p)-[:CONTAINS_SOURCE]->(d)
            """,
            params={"projectPath": str(self.project_path)},
        )
        logger.info("Linked :Project node to top-level source directories.")
        logger.info("--- Finished Pass: Establish Direct Source Hierarchy ---")

    def _ensure_source_directories_exist(self):
        """
        Creates fallback :Directory nodes for disk-backed :SourceFile nodes when the
        repository filesystem tree was not scanned into Neo4j.
        """
        query = """
        MATCH (sf:SourceFile)
        WHERE sf.absolute_path IS NOT NULL
          AND sf.absolute_path STARTS WITH $projectPath
        RETURN DISTINCT sf.absolute_path AS path
        """
        source_files = self.neo4j_manager.execute_read_query(
            query,
            params={"projectPath": str(self.project_path)},
        )

        if not source_files:
            logger.info("No project source files found for directory synthesis.")
            return

        anchor_markers = [
            "/src/main/java/",
            "/src/main/kotlin/",
            "/src/test/java/",
            "/src/test/kotlin/",
        ]

        dir_paths: set[str] = set()
        for row in source_files:
            source_path = row.get("path")
            if not source_path:
                continue

            anchor = str(self.project_path)
            for marker in anchor_markers:
                if marker in source_path:
                    anchor = source_path.split(marker, 1)[0] + marker.rstrip("/")
                    break

            current = Path(source_path).parent
            anchor_path = Path(anchor)
            while str(current).startswith(str(anchor_path)):
                dir_paths.add(str(current))
                if current == anchor_path:
                    break
                current = current.parent

        if not dir_paths:
            logger.info("No source directories needed to be synthesized.")
            return

        batch_size = 500
        ordered_paths = sorted(dir_paths)
        for start in range(0, len(ordered_paths), batch_size):
            chunk = ordered_paths[start : start + batch_size]
            self.neo4j_manager.execute_write_query(
                """
                UNWIND $paths AS path
                MERGE (d:Directory {absolute_path: path})
                ON CREATE SET d.fileName = path
                """,
                params={"paths": chunk},
            )

        logger.info(
            "Ensured fallback :Directory nodes exist for %s source directories.",
            len(ordered_paths),
        )
