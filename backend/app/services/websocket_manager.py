"""
WebSocket Connection Manager for real-time notifications.

Manages active WebSocket connections and broadcasts messages to all connected clients.
"""

from typing import List, Dict, Any
from fastapi import WebSocket
import json
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections for real-time notifications."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept and store a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📡 WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"📡 WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send a message to a specific client."""
        await websocket.send_json(message)
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected clients."""
        if not self.active_connections:
            print("📡 No active connections to broadcast to")
            return
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"📡 Failed to send to connection: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn)
        
        if self.active_connections:
            print(f"📡 Broadcast sent to {len(self.active_connections)} clients")
    
    async def broadcast_notification(
        self,
        notification_type: str,
        patient_id: str,
        patient_name: str,
        classification: str,
        confidence: float,
        findings: List[str],
        recommendation: str,
        severity: str = "info"
    ):
        """
        Broadcast a notification about analysis results.
        
        Args:
            notification_type: Type of notification (abnormality_detected, analysis_complete)
            patient_id: Patient ID
            patient_name: Patient name
            classification: AI classification result
            confidence: Confidence percentage
            findings: List of findings
            recommendation: Doctor recommendation
            severity: critical | warning | info | success
        """
        # Generate human-readable message based on severity and classification
        if severity == "critical":
            human_message = f"CRITICAL: {patient_name} shows {classification} with {confidence:.1f}% confidence. Immediate review required."
        elif severity == "warning":
            human_message = f"{patient_name} shows possible {classification} ({confidence:.1f}% confidence). Review recommended."
        elif severity == "success":
            human_message = f"{patient_name}'s scan appears normal ({confidence:.1f}% confidence)."
        else:
            human_message = f"Analysis complete for {patient_name}: {classification}"
        
        message = {
            "type": "scanner_notification",  # Unified type for all scanner notifications
            "data": {
                "patient_id": patient_id,
                "patient_name": patient_name,
                "classification": classification,
                "confidence": confidence,
                "findings": findings,
                "recommendation": recommendation,
                "severity": severity,
                "message": human_message,
                "notification_subtype": notification_type,  # Keep original type as subtype
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        await self.broadcast(message)
    
    async def broadcast_scanner_status(self, status: str, pending_count: int):
        """Broadcast scanner status update."""
        message = {
            "type": "scanner_status",
            "data": {
                "status": status,
                "pending_count": pending_count,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
        await self.broadcast(message)
    
    @property
    def connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)


# Global instance
manager = ConnectionManager()
