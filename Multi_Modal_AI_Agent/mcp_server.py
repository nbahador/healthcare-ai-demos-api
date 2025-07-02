import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
import websockets
from websockets.server import serve
from agent import MultiModalAgent

logger = logging.getLogger(__name__)

class MCPServer:
    """MCP Server for multi-modal AI agent"""
    
    def __init__(self, api_key: str):
        self.agent = MultiModalAgent(api_key)
        self.connections = set()
    
    async def register_connection(self, websocket):
        """Register a new client connection"""
        self.connections.add(websocket)
        logger.info(f"New MCP client connected. Total connections: {len(self.connections)}")
    
    async def unregister_connection(self, websocket):
        """Unregister a client connection"""
        self.connections.discard(websocket)
        logger.info(f"MCP client disconnected. Total connections: {len(self.connections)}")
    
    async def handle_message(self, websocket, message: str):
        """Handle incoming MCP messages"""
        try:
            data = json.loads(message)
            method = data.get("method")
            params = data.get("params", {})
            msg_id = data.get("id")
            
            if method == "analyze":
                # Process analysis request
                result = self.agent.process_request(**params)
                response = {
                    "id": msg_id,
                    "result": result
                }
            elif method == "ping":
                response = {
                    "id": msg_id,
                    "result": {"status": "pong"}
                }
            else:
                response = {
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Method '{method}' not found"}
                }
            
            await websocket.send(json.dumps(response))
            
        except json.JSONDecodeError:
            error_response = {
                "error": {"code": -32700, "message": "Parse error"}
            }
            await websocket.send(json.dumps(error_response))
        except Exception as e:
            error_response = {
                "id": data.get("id") if 'data' in locals() else None,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"}
            }
            await websocket.send(json.dumps(error_response))
    
    async def client_handler(self, websocket, path):
        """Handle individual client connections"""
        await self.register_connection(websocket)
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_connection(websocket)

def run_mcp_server(api_key: str, host: str = "0.0.0.0", port: int = 8001):
    """Run the MCP server"""
    server = MCPServer(api_key)
    
    start_server = serve(server.client_handler, host, port)
    
    logger.info(f"MCP Server starting on {host}:{port}")
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()