"""Redis client for NanoOps session state management."""

import json
import time
from typing import Any, Dict, List, Optional

import redis

from .exceptions import SessionExistsError, SessionNotFoundError


class RedisClient:
    """Redis client for managing session state."""

    def __init__(self, redis_url: str):
        """Initialize Redis client.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self.url = redis_url
        self.client = redis.from_url(redis_url, decode_responses=True)

    def _session_key(self, session_id: str) -> str:
        """Get session config key."""
        return f"{session_id}:session:config"

    def _component_key(
        self, session_id: str, component_type: str, component_id: str
    ) -> str:
        """Get component registry key."""
        return f"{session_id}:components:{component_type}:{component_id}"

    def _pg_key(self, session_id: str, pg_id: str) -> str:
        """Get placement group key."""
        return f"{session_id}:placement_groups:{pg_id}"

    # Session Operations

    def session_exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self.client.exists(self._session_key(session_id)) > 0

    def create_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """Create new session.

        Args:
            session_id: Unique session identifier
            data: Session metadata

        Raises:
            SessionExistsError: If session already exists
        """
        if self.session_exists(session_id):
            raise SessionExistsError(f"Session '{session_id}' already exists")

        # Add timestamp
        data["created_at"] = int(time.time())

        # Store as hash
        key = self._session_key(session_id)
        self.client.hset(key, mapping=self._serialize_dict(data))

    def get_session_config(self, session_id: str) -> Dict[str, Any]:
        """Get session configuration.

        Args:
            session_id: Session identifier

        Returns:
            Session configuration dict

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        if not self.session_exists(session_id):
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        key = self._session_key(session_id)
        data = self.client.hgetall(key)
        return self._deserialize_dict(data)

    def update_session_config(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update session configuration.

        Args:
            session_id: Session identifier
            updates: Configuration updates

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        if not self.session_exists(session_id):
            raise SessionNotFoundError(f"Session '{session_id}' not found")

        key = self._session_key(session_id)
        self.client.hset(key, mapping=self._serialize_dict(updates))

    # Component Operations

    def register_component(
        self,
        session_id: str,
        component_type: str,
        component_id: str,
        data: Dict[str, Any],
    ) -> None:
        """Register a spawned component.

        Args:
            session_id: Session identifier
            component_type: Component type (route, prefill, decode)
            component_id: Unique component ID (usually Ray job ID)
            data: Component metadata
        """
        data["spawned_at"] = int(time.time())

        key = self._component_key(session_id, component_type, component_id)
        self.client.hset(key, mapping=self._serialize_dict(data))

    def get_component_info(
        self, session_id: str, component_type: str, component_id: str
    ) -> Dict[str, Any]:
        """Get component information."""
        key = self._component_key(session_id, component_type, component_id)
        data = self.client.hgetall(key)
        return self._deserialize_dict(data) if data else {}

    def get_session_components(
        self, session_id: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all components for a session.

        Returns:
            Dict mapping component type to list of component info
        """
        pattern = f"{session_id}:components:*"
        keys = self.client.keys(pattern)

        components = {"route": [], "prefill": [], "decode": []}

        for key in keys:
            # Parse key: {session_id}:components:{type}:{id}
            parts = key.split(":")
            if len(parts) >= 4:
                component_type = parts[2]
                component_id = parts[3]

                data = self.client.hgetall(key)
                if data:
                    info = self._deserialize_dict(data)
                    info["component_id"] = component_id
                    info["component_type"] = component_type

                    if component_type in components:
                        components[component_type].append(info)

        return components

    def remove_component(self, session_id: str, component_id: str) -> bool:
        """Remove a component entry by its ID (Ray job ID).

        Searches all component types for the matching ID and deletes the key.

        Returns:
            True if a component was removed, False if not found.
        """
        for comp_type in ("route", "prefill", "decode"):
            key = self._component_key(session_id, comp_type, component_id)
            if self.client.exists(key):
                self.client.delete(key)
                return True
        return False

    # Placement Group Operations

    def register_placement_group(
        self, session_id: str, pg_id: str, component_type: str, num_gpus: int
    ) -> None:
        """Register a placement group.

        Args:
            session_id: Session identifier
            pg_id: Placement group ID (hex string)
            component_type: Component type (prefill, decode)
            num_gpus: Number of GPUs
        """
        data = {
            "component_type": component_type,
            "num_gpus": num_gpus,
            "strategy": "STRICT_PACK",
            "created_at": int(time.time()),
        }

        key = self._pg_key(session_id, pg_id)
        self.client.hset(key, mapping=self._serialize_dict(data))

    def get_placement_groups(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all placement groups for a session."""
        pattern = f"{session_id}:placement_groups:*"
        keys = self.client.keys(pattern)

        pgs = []
        for key in keys:
            # Extract PG ID from key
            pg_id = key.split(":")[-1]

            data = self.client.hgetall(key)
            if data:
                info = self._deserialize_dict(data)
                info["pg_id"] = pg_id
                pgs.append(info)

        return pgs

    # Session Cleanup

    def cleanup_session(self, session_id: str) -> int:
        """Delete all keys for a session.

        Returns:
            Number of keys deleted
        """
        pattern = f"{session_id}:*"
        keys = self.client.keys(pattern)

        if keys:
            return self.client.delete(*keys)
        return 0

    def list_sessions(self, include_stopped: bool = False) -> List[Dict[str, Any]]:
        """List all sessions.

        Args:
            include_stopped: Include sessions with status='stopped'

        Returns:
            List of session info dicts
        """
        # Find all session config keys
        pattern = "*:session:config"
        keys = self.client.keys(pattern)

        sessions = []
        for key in keys:
            # Extract session_id from key
            session_id = key.split(":")[0]

            data = self.client.hgetall(key)
            if data:
                info = self._deserialize_dict(data)
                info["session_id"] = session_id

                # Filter by status
                if not include_stopped and info.get("status") == "stopped":
                    continue

                # Add component count
                components = self.get_session_components(session_id)
                info["num_components"] = sum(len(v) for v in components.values())

                sessions.append(info)

        return sessions

    # Serialization Helpers

    def _serialize_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Serialize dict values to strings for Redis hash."""
        result = {}
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                result[key] = json.dumps(value)
            elif isinstance(value, bool):
                result[key] = "1" if value else "0"
            else:
                result[key] = str(value)
        return result

    def _deserialize_dict(self, data: Dict[str, str]) -> Dict[str, Any]:
        """Deserialize dict values from Redis hash."""
        result = {}
        for key, value in data.items():
            # Try to parse as JSON (for dicts/lists)
            if value.startswith("{") or value.startswith("["):
                try:
                    result[key] = json.loads(value)
                    continue
                except json.JSONDecodeError:
                    pass

            # Try to parse as int
            try:
                result[key] = int(value)
                continue
            except ValueError:
                pass

            # Try to parse as bool
            if value in ("0", "1"):
                result[key] = value == "1"
                continue

            # Keep as string
            result[key] = value

        return result
