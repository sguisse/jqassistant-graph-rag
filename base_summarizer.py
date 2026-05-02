import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from neo4j_manager import Neo4jManager
from node_summary_processor import NodeSummaryProcessor

logger = logging.getLogger(__name__)


class BaseSummarizer(ABC):
    """
    Abstract base class for summarization passes. Implements the Template Method
    pattern for processing batches of items.
    """

    def __init__(
        self,
        neo4j_manager: Neo4jManager,
        node_summary_processor: NodeSummaryProcessor,
        num_workers: int = 8,
    ):
        self.neo4j_manager = neo4j_manager
        self.node_summary_processor = node_summary_processor
        self.num_workers = num_workers
        logger.info(
            f"Initialized {self.__class__.__name__} with {self.num_workers} workers."
        )

    @abstractmethod
    def run(self) -> int:
        """The main entry point for the summarizer pass."""
        pass

    @abstractmethod
    def _get_update_query(self) -> str:
        """Returns the Cypher query for the batch update."""
        pass

    @abstractmethod
    def _get_processor_result(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Abstract method for subclasses to call the correct NodeSummaryProcessor method.
        This is the "primitive operation" in the Template Method pattern.
        """
        pass

    def _prepare_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optional hook for subclasses to modify an item before processing.
        By default, it does nothing.
        """
        return item

    def _handle_result(
        self, result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Handles the result from the NodeSummaryProcessor, updating caches and runtime status.
        """
        if not result:
            return None

        node_id = result.get("id")
        if not node_id:
            logger.warning(
                "Skipping %s result without entity_id: %s",
                self.__class__.__name__,
                result,
            )
            return None

        status = result["status"]

        cache_data = {}
        if "summary" in result:
            cache_data["summary"] = result["summary"]
        if "code_analysis" in result:
            cache_data["code_analysis"] = result["code_analysis"]
        if "code_hash" in result:
            cache_data["code_hash"] = result["code_hash"]

        if status in ["regenerated", "restored"]:
            self.node_summary_processor.cache_manager.update_node_cache(
                node_id, cache_data
            )
            if status == "regenerated":
                self.node_summary_processor.cache_manager.set_runtime_status(
                    node_id, "regenerated"
                )
            return result
        elif status == "unchanged":
            self.node_summary_processor.cache_manager.update_node_cache(
                node_id, cache_data
            )

        return None

    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Prepares the item and computes its raw processor result.

        Cache updates are intentionally NOT applied here: applying them in the
        worker thread would make the batch behavior order-dependent and
        nondeterministic because sibling items could observe partial cache
        updates from other in-flight items.
        """
        prepared_item = self._prepare_item(item)
        return self._get_processor_result(prepared_item)

    def process_batch(self, items_to_process: List[Dict[str, Any]]) -> int:
        """
        Processes a given list of items in parallel using the template method.
        """
        if not items_to_process:
            return 0

        class_name = self.__class__.__name__
        logger.info(
            f"Processing batch of {len(items_to_process)} items for {class_name}."
        )

        updates = []
        raw_results: list[Optional[Dict[str, Any]]] = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._process_item, item): item
                for item in items_to_process
            }

            for future in tqdm(
                as_completed(futures),
                total=len(items_to_process),
                desc=f"Processing {class_name} batch",
            ):
                try:
                    raw_results.append(future.result())
                except Exception as e:
                    item = futures[future]
                    logger.error(
                        f"Error processing item {item.get('id', 'N/A')} in {class_name}: {e}",
                        exc_info=True,
                    )

        # Apply cache/runtime updates ONLY after the full batch has completed so
        # every item in the batch saw the same pre-batch cache state.
        for raw_result in raw_results:
            update_data = self._handle_result(raw_result)
            if update_data:
                updates.append(update_data)

        if not updates:
            logger.info(
                f"No database updates needed for this batch in {class_name} (all items up-to-date)."
            )
            return 0

        update_query = self._get_update_query()
        summary_counters = self.neo4j_manager.execute_write_query(
            update_query, params={"updates": updates}
        )

        properties_set = summary_counters.properties_set if summary_counters else 0
        logger.info(
            f"Batch complete for {class_name}. Updated {properties_set} properties."
        )
        return properties_set
