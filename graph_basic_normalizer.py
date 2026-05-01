import logging
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

        # Set absolute_path for top-level :Directory nodes, including both
        # Artifact:Directory nodes (scanned with java:classpath:: prefix) and
        # non-Artifact:Directory nodes (scanned directly as directories).
        # We need set the absolute_path for the top-level :Directory nodes themselves,
        # and also for the files and directories contained within them.
        # We jQAssistant scans a directory, the entry node is a :Directory node,
        # and its fileName property is the absolute path of the directory.
        # This entry node is the root of the directory tree.
        # It _directly_ :CONTAINS all of the :File and :Directory (also a :File) nodes in the tree.
        # The containership does not cross Artifact boundaries
        # (i.e., the root node does not :CONTAINS the nodes under an :Artifact:Jar file node).
        # This makes it easy to find all the descendants of the root node by a single step :CONTAINS.
        # The fileName of the nodes within the directory tree is the relative path from the entry node.
        # So we need to concatenate the parent's absolute_path with the children nodes' fileName for the absolute_path of the children nodes.
        directory_tree_query = """
        MATCH (e:Directory)
        WHERE NOT EXISTS { (:Directory)-[:CONTAINS]->(e) }
        SET e.absolute_path = e.fileName
        WITH e
        MATCH (e)-[:CONTAINS]->(c:File)
        SET c.absolute_path = e.absolute_path + c.fileName
        RETURN count(e) + count(c) AS paths_normalized
        """
        dir_tree_result = self.neo4j_manager.execute_write_query(directory_tree_query)
        dir_tree_props_set = dir_tree_result.properties_set
        logger.info(
            f"Set 'absolute_path' for {dir_tree_props_set} Directory nodes and their contained files."
        )

        # continue with artifact and contained path normalization

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
        WHERE e.fileName IS NOT NULL AND f.fileName IS NOT NULL
        SET f.absolute_path = e.fileName + f.fileName
        RETURN count(f) AS paths_normalized
        """
        contained_result = self.neo4j_manager.execute_write_query(contained_query)
        contained_props_set = contained_result.properties_set
        logger.info(
            f"Set 'absolute_path' for {contained_props_set} contained File/Directory nodes."
        )

        # Step 2: Set absolute_path for TOP-LEVEL non-Artifact:Directory nodes ONLY.
        # 2.1. If jQAssistant scans a directory without using java:classpath:: prefix,
        # User may scan a directory with source files only in this way.
        # In this case, the top-level nodes will be :Directory nodes, not Artifact:Directory nodes.
        # IMPORTANT: Only target top-level (non-contained) directories here.
        # Applying this to ALL directories overwrites subdirectory paths set in directory_tree_query
        # (which correctly set them to parent.absolute_path + child.fileName) with the raw
        # relative fileName, causing CONTAINS_SOURCE linking to break.
        non_artifact_query = """
        MATCH (e:Directory)
        WHERE e.fileName IS NOT NULL AND NOT e:Artifact
          AND NOT EXISTS { (:Directory)-[:CONTAINS]->(e) }
        SET e.absolute_path = e.fileName
        RETURN count(e) AS paths_normalized
        """
        non_artifact_result = self.neo4j_manager.execute_write_query(non_artifact_query)
        non_artifact_props_set = non_artifact_result.properties_set
        logger.info(
            f"Set 'absolute_path' for {non_artifact_props_set} top-level non-Artifact Directory nodes."
        )

        # Step 3: Propagate absolute_path from parent to child nodes, level by level.
        # directory_tree_query only covers top-level → direct children (depth 1).
        # This loop handles deeper nesting (grandchildren, etc.) by repeatedly matching
        # parent nodes that already have absolute_path and setting it on their children.
        logger.info("Propagating absolute_path through directory tree (all depths)...")
        total_propagated = 0
        for _level in range(20):  # cap at 20 levels — deep enough for any real project
            propagate_result = self.neo4j_manager.execute_write_query(
                """
                MATCH (parent:Directory)-[:CONTAINS]->(child:File)
                WHERE parent.absolute_path IS NOT NULL
                  AND child.absolute_path IS NULL
                  AND child.fileName IS NOT NULL
                SET child.absolute_path = parent.absolute_path + child.fileName
                RETURN count(child) AS updated
                """
            )
            n = propagate_result.properties_set if propagate_result else 0
            if n == 0:
                break
            total_propagated += n
        logger.info(f"Propagated absolute_path to {total_propagated} additional nodes.")

        logger.info("--- Finished Pass: Add Absolute Paths ---")
        # Ensure disk SourceFile nodes exist (fallback) in case jQAssistant filesystem
        # scanner did not populate them.  Run AFTER propagation so that MERGE on
        # absolute_path finds already-enriched jqAssistant File nodes instead of
        # creating duplicate nodes.
        try:
            self.create_disk_source_files()
        except Exception:
            logger.exception(
                "Failed to create disk-sourced SourceFile nodes (fallback)"
            )

    def label_source_files(self):
        """
        Identifies and labels :File nodes that represent Java or Kotlin
        source code files as :SourceFile.
        This pass relies on 'absolute_path' having been set previously.
        """
        logger.info("--- Starting Pass: Label Source Files ---")

        # Primary: label files which already have absolute_path set
        query = """
        MATCH (f:File)
        WHERE f.absolute_path IS NOT NULL
        AND (f.absolute_path ENDS WITH '.java' OR f.absolute_path ENDS WITH '.kt')
        SET f:SourceFile
        RETURN count(f) AS source_files_labeled
        """
        result = self.neo4j_manager.execute_write_query(query)
        labels_added = getattr(result, "labels_added", 0)

        # Secondary: some scans store the path in fileName but not absolute_path.
        # Normalize by copying fileName -> absolute_path and labeling those files.
        secondary_query = """
        MATCH (f:File)
        WHERE (f.absolute_path IS NULL OR f.absolute_path = NULL)
          AND f.fileName IS NOT NULL
          AND (f.fileName ENDS WITH '.java' OR f.fileName ENDS WITH '.kt')
        SET f.absolute_path = f.fileName, f:SourceFile
        RETURN count(f) AS source_files_labeled_secondary
        """
        sec_result = self.neo4j_manager.execute_write_query(secondary_query)
        sec_added = getattr(sec_result, "labels_added", 0)

        total = (labels_added or 0) + (sec_added or 0)
        logger.info(f"Labeled {total} files as :SourceFile.")
        logger.info("--- Finished Pass: Label Source Files ---")

    def create_disk_source_files(
        self, repo_root: str | None = None, batch_size: int = 200
    ):
        """
        Create :File and :SourceFile nodes from the repository filesystem (src/main/java).
        This is a fallback for environments where the jQAssistant filesystem scanner
        did not run or did not populate SourceFile nodes.
        """
        logger.info("--- Starting Pass: Create Disk SourceFile nodes (fallback) ---")
        from pathlib import Path

        root = Path(repo_root) if repo_root else Path.cwd()
        # If cwd is inside the tool-graph-rag folder, walk up to find project root (pom.xml)
        p = root
        for _ in range(8):
            if (p / "pom.xml").exists():
                root = p
                break
            p = p.parent
        src = root / "src" / "main" / "java"
        if not src.exists():
            logger.info(
                f"Source dir not found: {src} — skipping disk SourceFile creation."
            )
            return 0

        paths: list[str] = []
        for p in src.rglob("*.java"):
            paths.append(str(p))
        for p in src.rglob("*.kt"):
            paths.append(str(p))

        if not paths:
            logger.info("No Java/Kotlin source files found on disk — skipping.")
            return 0

        total_created = 0
        # batch INSERTs via UNWIND to avoid overly large parameter lists.
        # MERGE on absolute_path (not fileName) so that jqAssistant-created File nodes
        # (which have their absolute_path set by the propagation loop above) are found and
        # simply labelled :SourceFile instead of creating a duplicate node.
        for i in range(0, len(paths), batch_size):
            chunk = paths[i : i + batch_size]
            cypher = (
                "UNWIND $paths AS p\n"
                "MERGE (f:File {absolute_path: p})\n"
                "  ON CREATE SET f.fileName = p, f:SourceFile\n"
                "  ON MATCH  SET f:SourceFile\n"
                "RETURN count(f) AS created"
            )
            res = self.neo4j_manager.execute_write_query(
                cypher, params={"paths": chunk}
            )
            # some execute_write_query implementations return properties_set or labels_added
            created = getattr(res, "properties_set", None) or getattr(
                res, "labels_added", None
            )
            try:
                # if the driver returned a result object with records
                if created is None and hasattr(res, "records"):
                    created = sum(int(r[0]) for r in res.records)
            except Exception:
                created = None

            if isinstance(created, int):
                total_created += created

        logger.info(f"Created or updated {total_created} disk SourceFile nodes.")
        logger.info("--- Finished Pass: Create Disk SourceFile nodes (fallback) ---")
        return total_created
