"""KataGo player using async HTTP to a remote analysis server.

Connects to an already-running KataGo analysis server via HTTP endpoint.
The server should be started separately (via PBS jobs managed by manage_servers.sh).
"""

import asyncio
import time
from typing import List, Optional

import aiohttp

from eval.players.base import Player


class KataGoPlayer(Player):
    """KataGo player using HTTP to communicate with remote analysis server."""
    
    def __init__(
        self,
        endpoint: str,
        name: Optional[str] = None
    ):
        """Initialize KataGo player.
        
        Args:
            endpoint: URL of already-running KataGo server (e.g., "http://hopper-34:9200")
            name: Player name
        """
        self._base_url = endpoint.rstrip("/")
        player_name = name or "katago"
        
        super().__init__(name=player_name)
        
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def requires_logging(self) -> bool:
        """KataGo players don't need LLM-style logging."""
        return False
    
    async def start(self):
        """Start the player by connecting to the remote server."""
        if self._session is not None:
            return
        
        self._session = aiohttp.ClientSession()
        print(f"Connecting to remote KataGo server: {self._base_url}")
        await self._wait_for_server(timeout=30)
        print(f"Connected to KataGo server: {self._base_url}")
    
    async def _wait_for_server(self, timeout: float = 30):
        """Wait for the server to become available."""
        start_time = time.time()
        last_error = None
        
        while time.time() - start_time < timeout:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self._base_url}/health", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            return
            except Exception as e:
                last_error = e
            
            await asyncio.sleep(1)
        
        raise RuntimeError(f"KataGo server at {self._base_url} not reachable within {timeout}s: {last_error}")
    
    async def stop(self):
        """Stop the player (disconnect from server)."""
        if self._session is not None:
            await self._session.close()
            self._session = None
        print(f"Disconnected from KataGo server: {self.name}")
    
    async def get_move(
        self,
        move_history: List[List[str]],
        rules: str,
        komi: float,
        color: str,
        win_rate: Optional[float] = None
    ) -> str:
        """Get move from KataGo server via HTTP."""
        assert self._session is not None, "Player not started"
        
        payload = {
            "moves": move_history,
            "rules": rules,
            "komi": komi,
            "color": color,
            "board_size": 19
        }
        
        try:
            async with self._session.post(
                f"{self._base_url}/move",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    print(f"KataGo server error: {text}")
                    return ""
                
                data = await resp.json()
                return data.get("move", "")
        
        except Exception as e:
            print(f"Error getting move from KataGo: {e}")
            return ""
    
    async def get_win_rate(
        self,
        move_history: List[List[str]],
        rules: str,
        komi: float,
        color: str
    ) -> Optional[float]:
        """Get win rate from KataGo server."""
        assert self._session is not None, "Player not started"
        
        payload = {
            "moves": move_history,
            "rules": rules,
            "komi": komi,
            "color": color,
            "board_size": 19
        }
        
        try:
            async with self._session.post(
                f"{self._base_url}/analyze_position",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    return None
                
                data = await resp.json()
                return data.get("win_rate")
        
        except Exception:
            return None
