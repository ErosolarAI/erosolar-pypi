"""Vector store for efficient similarity search over large tool collections."""
from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class VectorEntry:
    """Single entry in the vector store."""
    id: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    Efficient vector storage and similarity search for tool embeddings.

    Features:
    - Numpy-based for fast similarity computation
    - Persistent storage support
    - Batch operations
    - Multiple similarity metrics
    """

    def __init__(
        self,
        dimension: int = 1536,  # text-embedding-3-small dimension
        metric: str = "cosine",
        cache_dir: Optional[str] = None,
    ):
        self.dimension = dimension
        self.metric = metric
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Storage
        self.entries: Dict[str, VectorEntry] = {}
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.id_to_index: Dict[str, int] = {}
        self.index_to_id: Dict[int, str] = {}

        # Stats
        self.stats = {
            "searches": 0,
            "avg_search_time_ms": 0,
            "total_entries": 0,
            "last_rebuild": None,
        }

        # Load from cache if available
        if self.cache_dir and self.cache_dir.exists():
            self._load_from_cache()

    def add(self, id: str, embedding: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Add a single vector to the store."""
        if len(embedding) != self.dimension:
            raise ValueError(f"Embedding dimension {len(embedding)} != expected {self.dimension}")

        self.entries[id] = VectorEntry(
            id=id,
            embedding=np.array(embedding, dtype=np.float32),
            metadata=metadata or {}
        )

        # Mark matrix for rebuild
        self.embeddings_matrix = None
        self.stats["total_entries"] = len(self.entries)

    def add_batch(self, items: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> None:
        """Add multiple vectors efficiently."""
        for id, embedding, metadata in items:
            if len(embedding) != self.dimension:
                raise ValueError(f"Embedding dimension {len(embedding)} != expected {self.dimension}")

            self.entries[id] = VectorEntry(
                id=id,
                embedding=np.array(embedding, dtype=np.float32),
                metadata=metadata or {}
            )

        # Mark matrix for rebuild
        self.embeddings_matrix = None
        self.stats["total_entries"] = len(self.entries)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors.

        Returns list of (id, similarity_score, metadata) tuples.
        """
        start_time = time.perf_counter()

        if not self.entries:
            return []

        # Rebuild matrix if needed
        if self.embeddings_matrix is None:
            self._rebuild_matrix()

        # Normalize query embedding for cosine similarity
        query_embedding = np.array(query_embedding, dtype=np.float32)
        if self.metric == "cosine":
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm

        # Compute similarities
        if self.metric == "cosine":
            similarities = np.dot(self.embeddings_matrix, query_embedding)
        elif self.metric == "euclidean":
            # Convert to similarity (inverse of distance)
            distances = np.linalg.norm(self.embeddings_matrix - query_embedding, axis=1)
            similarities = 1.0 / (1.0 + distances)
        elif self.metric == "dot":
            similarities = np.dot(self.embeddings_matrix, query_embedding)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Apply metadata filter if provided
        valid_indices = np.arange(len(similarities))
        if filter_metadata:
            valid_indices = self._filter_by_metadata(filter_metadata)
            if len(valid_indices) == 0:
                return []
            similarities = similarities[valid_indices]

        # Apply threshold if provided
        if threshold is not None:
            mask = similarities >= threshold
            valid_indices = valid_indices[mask]
            similarities = similarities[mask]
            if len(similarities) == 0:
                return []

        # Get top-k results
        k = min(top_k, len(similarities))
        if k == 0:
            return []

        # Use argpartition for efficiency with large arrays
        if len(similarities) > k * 10:  # Only use argpartition for large arrays
            top_indices = np.argpartition(similarities, -k)[-k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1][:k]

        # Build results
        results = []
        for idx in top_indices:
            actual_idx = valid_indices[idx] if filter_metadata or threshold else idx
            id = self.index_to_id[actual_idx]
            score = float(similarities[idx])
            metadata = self.entries[id].metadata
            results.append((id, score, metadata))

        # Update stats
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.stats["searches"] += 1
        self.stats["avg_search_time_ms"] = (
            (self.stats["avg_search_time_ms"] * (self.stats["searches"] - 1) + elapsed_ms)
            / self.stats["searches"]
        )

        return results

    def search_batch(
        self,
        query_embeddings: List[np.ndarray],
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> List[List[Tuple[str, float, Dict]]]:
        """Batch search for multiple queries."""
        results = []
        for query_embedding in query_embeddings:
            results.append(
                self.search(query_embedding, top_k, filter_metadata, threshold)
            )
        return results

    def get(self, id: str) -> Optional[VectorEntry]:
        """Get a specific entry by ID."""
        return self.entries.get(id)

    def remove(self, id: str) -> bool:
        """Remove an entry by ID."""
        if id in self.entries:
            del self.entries[id]
            self.embeddings_matrix = None
            self.stats["total_entries"] = len(self.entries)
            return True
        return False

    def clear(self) -> None:
        """Clear all entries."""
        self.entries.clear()
        self.embeddings_matrix = None
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.stats["total_entries"] = 0

    def _rebuild_matrix(self) -> None:
        """Rebuild the embeddings matrix for efficient search."""
        if not self.entries:
            self.embeddings_matrix = np.empty((0, self.dimension), dtype=np.float32)
            return

        # Build matrix and mappings
        ids = list(self.entries.keys())
        embeddings = []

        for i, id in enumerate(ids):
            entry = self.entries[id]
            embedding = entry.embedding

            # Normalize for cosine similarity
            if self.metric == "cosine":
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm

            embeddings.append(embedding)
            self.id_to_index[id] = i
            self.index_to_id[i] = id

        self.embeddings_matrix = np.vstack(embeddings).astype(np.float32)
        self.stats["last_rebuild"] = time.time()

    def _filter_by_metadata(self, filter_metadata: Dict[str, Any]) -> np.ndarray:
        """Get indices of entries matching metadata filter."""
        valid_indices = []

        for idx, id in self.index_to_id.items():
            entry = self.entries[id]
            match = True

            for key, value in filter_metadata.items():
                if key not in entry.metadata:
                    match = False
                    break

                # Handle list values (e.g., categories)
                if isinstance(value, list):
                    if not any(v in entry.metadata.get(key, []) for v in value):
                        match = False
                        break
                elif entry.metadata.get(key) != value:
                    match = False
                    break

            if match:
                valid_indices.append(idx)

        return np.array(valid_indices, dtype=int)

    def save_to_cache(self) -> None:
        """Save the vector store to disk."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Save entries
        entries_file = self.cache_dir / "vector_store_entries.pkl"
        with open(entries_file, "wb") as f:
            pickle.dump(self.entries, f)

        # Save stats
        stats_file = self.cache_dir / "vector_store_stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f)

    def _load_from_cache(self) -> None:
        """Load the vector store from disk."""
        entries_file = self.cache_dir / "vector_store_entries.pkl"
        stats_file = self.cache_dir / "vector_store_stats.json"

        if entries_file.exists():
            with open(entries_file, "rb") as f:
                self.entries = pickle.load(f)
            self.embeddings_matrix = None  # Will rebuild on first search

        if stats_file.exists():
            with open(stats_file, "r") as f:
                self.stats.update(json.load(f))

        self.stats["total_entries"] = len(self.entries)

    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self.stats,
            "memory_mb": self._estimate_memory_usage() / (1024 * 1024),
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in bytes."""
        # Embeddings matrix
        matrix_size = 0
        if self.embeddings_matrix is not None:
            matrix_size = self.embeddings_matrix.nbytes

        # Entries (rough estimate)
        entries_size = len(self.entries) * (
            self.dimension * 4 +  # embedding (float32)
            100  # metadata overhead estimate
        )

        return matrix_size + entries_size


class HierarchicalVectorStore:
    """
    Hierarchical vector store for even larger scale.

    Uses clustering to create a two-level index:
    1. Cluster centers for coarse search
    2. Vectors within clusters for fine search
    """

    def __init__(
        self,
        dimension: int = 1536,
        n_clusters: int = 100,
        metric: str = "cosine",
        cache_dir: Optional[str] = None,
    ):
        self.dimension = dimension
        self.n_clusters = n_clusters
        self.metric = metric
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Cluster stores
        self.cluster_centers: Optional[np.ndarray] = None
        self.clusters: Dict[int, VectorStore] = {}

        # Global ID mapping
        self.id_to_cluster: Dict[str, int] = {}

    def add_batch(self, items: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> None:
        """Add items and rebuild clusters if needed."""
        # For now, simple implementation - can be optimized with incremental clustering
        if not self.clusters:
            self._initialize_clusters(items)
        else:
            # Add to nearest clusters
            for id, embedding, metadata in items:
                cluster_id = self._find_nearest_cluster(embedding)
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = VectorStore(
                        dimension=self.dimension,
                        metric=self.metric
                    )
                self.clusters[cluster_id].add(id, embedding, metadata)
                self.id_to_cluster[id] = cluster_id

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        n_probe: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float, Dict]]:
        """
        Hierarchical search:
        1. Find top n_probe clusters
        2. Search within those clusters
        3. Merge and return top_k results
        """
        if not self.clusters:
            return []

        # Find nearest clusters
        nearest_clusters = self._find_nearest_clusters(query_embedding, n_probe)

        # Search within clusters
        all_results = []
        for cluster_id in nearest_clusters:
            if cluster_id in self.clusters:
                cluster_results = self.clusters[cluster_id].search(
                    query_embedding,
                    top_k * 2,  # Get more results for merging
                    filter_metadata,
                    threshold
                )
                all_results.extend(cluster_results)

        # Sort by score and return top_k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]

    def _initialize_clusters(self, items: List[Tuple[str, np.ndarray, Optional[Dict]]]) -> None:
        """Initialize clusters using k-means++."""
        if len(items) < self.n_clusters:
            # Too few items, use single cluster
            self.clusters[0] = VectorStore(dimension=self.dimension, metric=self.metric)
            for id, embedding, metadata in items:
                self.clusters[0].add(id, embedding, metadata)
                self.id_to_cluster[id] = 0
            self.cluster_centers = np.mean([item[1] for item in items], axis=0, keepdims=True)
        else:
            # Simple k-means implementation (can be replaced with sklearn if available)
            embeddings = np.array([item[1] for item in items])
            self.cluster_centers = self._kmeans_simple(embeddings, self.n_clusters)

            # Assign items to clusters
            for id, embedding, metadata in items:
                cluster_id = self._find_nearest_cluster(embedding)
                if cluster_id not in self.clusters:
                    self.clusters[cluster_id] = VectorStore(
                        dimension=self.dimension,
                        metric=self.metric
                    )
                self.clusters[cluster_id].add(id, embedding, metadata)
                self.id_to_cluster[id] = cluster_id

    def _kmeans_simple(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple k-means implementation."""
        n_samples = len(embeddings)

        # Initialize centers with k-means++
        centers = []
        center_indices = []

        # First center is random
        first_idx = np.random.randint(n_samples)
        centers.append(embeddings[first_idx])
        center_indices.append(first_idx)

        # Select remaining centers
        for _ in range(1, n_clusters):
            # Compute distances to nearest center
            distances = np.full(n_samples, np.inf)
            for center in centers:
                dists = np.linalg.norm(embeddings - center, axis=1)
                distances = np.minimum(distances, dists)

            # Probability proportional to squared distance
            probabilities = distances ** 2
            probabilities /= probabilities.sum()

            # Select next center
            next_idx = np.random.choice(n_samples, p=probabilities)
            centers.append(embeddings[next_idx])
            center_indices.append(next_idx)

        return np.array(centers)

    def _find_nearest_cluster(self, embedding: np.ndarray) -> int:
        """Find the nearest cluster center."""
        if self.cluster_centers is None:
            return 0

        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        return int(np.argmin(distances))

    def _find_nearest_clusters(self, embedding: np.ndarray, n_probe: int) -> List[int]:
        """Find the n_probe nearest cluster centers."""
        if self.cluster_centers is None:
            return [0]

        distances = np.linalg.norm(self.cluster_centers - embedding, axis=1)
        n_probe = min(n_probe, len(distances))
        nearest_indices = np.argpartition(distances, n_probe - 1)[:n_probe]
        return nearest_indices.tolist()