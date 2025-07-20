from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from haiku.rag.store.engine import Store

T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """Base repository interface for database operations."""

    def __init__(self, store: Store):
        self.store = store

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity in the database."""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: int) -> T | None:
        """Get an entity by its ID."""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass

    @abstractmethod
    async def delete(self, entity_id: int) -> bool:
        """Delete an entity by its ID."""
        pass

    @abstractmethod
    async def list_all(
        self, limit: int | None = None, offset: int | None = None
    ) -> list[T]:
        """List all entities with optional pagination."""
        pass
