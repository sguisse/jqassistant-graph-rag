import logging
from neo4j_manager import Neo4jManager

logger = logging.getLogger(__name__)


class GraphEntitySetter:
    """
    Handles the final phase of graph normalization: labeling all relevant nodes
    as :Entity and assigning them a stable, unique entity_id.
    """

    def __init__(self, neo4j_manager: Neo4jManager):
        self.neo4j_manager = neo4j_manager
        logger.info("Initialized GraphEntitySetter.")

    def create_entities_and_stable_ids(self):
        """
        Creates a stable, unique 'entity_id' for all relevant nodes and
        labels them as :Entity. This pass is critical for caching and
        dependency tracking.
        """
        logger.info("--- Starting Pass: Create Entities and Stable IDs ---")

        # 1. Create uniqueness constraint
        self.neo4j_manager.execute_write_query(
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE"
        )
        logger.info("Ensured :Entity(entity_id) uniqueness constraint exists.")

        # 2. Generate entity_id for :Project
        # The NOT EXISTS guard prevents a duplicate-entity_id ConstraintValidationFailed
        # when stale Project nodes (from earlier runs with a different name) survive.
        self.neo4j_manager.execute_write_query(
            """
            MATCH (p:Project)
            WHERE p.entity_id IS NULL
            WITH p, apoc.util.md5(["Project://", p.absolute_path]) AS eid
            WHERE NOT EXISTS { (e:Entity {entity_id: eid}) }
            SET p:Entity, p.entity_id = eid
            """
        )
        logger.info("Generated entity_id for :Project node.")

        # 3. (NEW) Generate entity_id for source tree nodes not part of any artifact
        self.neo4j_manager.execute_write_query(
            """
            MATCH (demotedRoot:Directory)
            WHERE demotedRoot.fileName = demotedRoot.absolute_path AND NOT demotedRoot:Artifact
            MATCH (descendant:File)
            WHERE descendant.absolute_path STARTS WITH demotedRoot.absolute_path
              AND NOT EXISTS { (:Artifact)-[:CONTAINS]->(descendant) }
              AND descendant.entity_id IS NULL
            WITH descendant, demotedRoot, apoc.util.md5([demotedRoot.fileName, descendant.fileName]) AS eid
            SET descendant:Entity, descendant.entity_id = eid
            """
        )
        logger.info("Generated entity_id for source tree nodes.")

        # 4. Generate entity_id for :Artifact
        self.neo4j_manager.execute_write_query(
            """
            MATCH (a:Artifact)
            WHERE a.fileName IS NOT NULL AND a.entity_id IS NULL
            WITH a, apoc.util.md5([a.fileName]) AS eid
            SET a:Entity, a.entity_id = eid
            """
        )
        logger.info("Generated entity_id for :Artifact nodes.")

        # 5. Generate entity_id for file-system-like nodes WITHIN artifacts
        self.neo4j_manager.execute_write_query(
            """
            MATCH (a:Artifact)-[:CONTAINS]->(n)
            WHERE (n:File OR n:Directory)
            AND n.fileName IS NOT NULL AND a.fileName IS NOT NULL AND n.entity_id IS NULL
            WITH n, a, apoc.util.md5([a.fileName, n.fileName]) AS eid
            SET n:Entity, n.entity_id = eid
            """
        )
        logger.info("Generated entity_id for file-system-like nodes within artifacts.")

        # 5b. Generate entity_id for remaining :Directory nodes (e.g. synthesized source dirs)
        self.neo4j_manager.execute_write_query(
            """
            MATCH (d:Directory)
            WHERE coalesce(d.absolute_path, d.fileName) IS NOT NULL
              AND d.entity_id IS NULL
            WITH coalesce(d.absolute_path, d.fileName) AS stable_path, collect(d) AS dirs
            WITH stable_path, head(dirs) AS d, apoc.util.md5(["Directory://", stable_path]) AS eid
            WHERE NOT EXISTS { (e:Entity {entity_id: eid}) }
            SET d:Entity, d.entity_id = eid
            """
        )
        logger.info("Generated entity_id for remaining :Directory nodes.")

        # 5c. Generate entity_id for remaining :File/:SourceFile nodes (e.g. disk-backed source files)
        self.neo4j_manager.execute_write_query(
            """
            MATCH (f:File)
            WHERE coalesce(f.absolute_path, f.fileName) IS NOT NULL
              AND f.entity_id IS NULL
            WITH coalesce(f.absolute_path, f.fileName) AS stable_path, collect(f) AS files
            WITH stable_path, head(files) AS f, apoc.util.md5(["File://", stable_path]) AS eid
            WHERE NOT EXISTS { (e:Entity {entity_id: eid}) }
            SET f:Entity, f.entity_id = eid
            """
        )
        logger.info("Generated entity_id for remaining :File/:SourceFile nodes.")

        # 6. Generate entity_id for :Member nodes from any declared source or artifact type
        self.neo4j_manager.execute_write_query(
            """
            MATCH (t:Type)-[:DECLARES]->(m)
            WHERE (m:Member OR m:Method OR m:Field)
              AND t.entity_id IS NOT NULL
              AND m.entity_id IS NULL
            WITH t, m, coalesce(m.signature, m.name, m.fileName) AS member_key
            WHERE member_key IS NOT NULL
            WITH t.entity_id AS type_id, member_key, collect(m) AS members
            WITH type_id, member_key, head(members) AS m,
                 apoc.util.md5(["Member://", type_id, member_key]) AS eid
            WHERE NOT EXISTS { (e:Entity {entity_id: eid}) }
            SET m:Entity, m.entity_id = eid
            """
        )
        logger.info("Generated entity_id for :Member nodes.")
        logger.info("--- Finished Pass: Create Entities and Stable IDs ---")
