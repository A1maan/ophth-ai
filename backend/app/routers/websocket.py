"""
WebSocket router for real-time notifications.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.websocket_manager import manager
from app.services.scanner import is_scanner_running, trigger_manual_scan

router = APIRouter()


@router.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time notifications.
    
    Clients connect here to receive live updates about:
    - Abnormality detections
    - Analysis completions
    - Scanner status updates
    """
    await manager.connect(websocket)
    
    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "data": {
                "message": "Connected to Ophth-AI notification stream",
                "scanner_running": is_scanner_running()
            }
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                data = await websocket.receive_text()
                
                # Handle client commands
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data == "status":
                    await websocket.send_json({
                        "type": "status",
                        "data": {
                            "scanner_running": is_scanner_running(),
                            "connections": manager.connection_count
                        }
                    })
                elif data == "trigger_scan":
                    # Allow manual scan trigger from client
                    await trigger_manual_scan()
                    
            except WebSocketDisconnect:
                break
                
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


@router.get("/ws/status")
async def get_websocket_status():
    """Get WebSocket connection status (HTTP endpoint for debugging)."""
    return {
        "active_connections": manager.connection_count,
        "scanner_running": is_scanner_running()
    }
