from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.models import Notification
from app.schemas.schemas import (
    NotificationCreate,
    NotificationResponseCamel,
    MessageResponse
)
from app.services.realtime import notification_manager

router = APIRouter()


def notification_to_camel(notification: Notification) -> dict:
    """Convert SQLAlchemy Notification to camelCase dict for frontend."""
    return {
        "id": notification.id,
        "patientId": notification.patient_id,
        "message": notification.message,
        "timestamp": notification.timestamp.isoformat() if notification.timestamp else None,
        "read": notification.read,
        "type": notification.type.value if notification.type else "info"
    }


@router.get("/", response_model=List[NotificationResponseCamel])
async def get_notifications(
    skip: int = 0,
    limit: int = 50,
    unread_only: bool = False,
    db: Session = Depends(get_db)
):
    """Get all notifications."""
    query = db.query(Notification)
    
    if unread_only:
        query = query.filter(Notification.read == False)
    
    notifications = query.order_by(Notification.timestamp.desc()).offset(skip).limit(limit).all()
    return [notification_to_camel(n) for n in notifications]


@router.get("/unread-count")
async def get_unread_count(db: Session = Depends(get_db)):
    """Get count of unread notifications."""
    count = db.query(Notification).filter(Notification.read == False).count()
    return {"count": count}


@router.post("/", response_model=NotificationResponseCamel, status_code=status.HTTP_201_CREATED)
async def create_notification(
    notification_data: NotificationCreate,
    db: Session = Depends(get_db)
):
    """Create a new notification."""
    from app.models.models import NotificationType
    
    notification = Notification(
        patient_id=notification_data.patient_id,
        message=notification_data.message,
        type=NotificationType(notification_data.type.value)
    )
    
    db.add(notification)
    db.commit()
    db.refresh(notification)

    response = notification_to_camel(notification)

    # Push to any live WebSocket subscribers for real-time UI updates
    await notification_manager.broadcast({
        "event": "notification",
        "data": response
    })
    
    return response


@router.patch("/{notification_id}/read")
async def mark_notification_read(notification_id: str, db: Session = Depends(get_db)):
    """Mark a notification as read."""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notification with id {notification_id} not found"
        )
    
    notification.read = True
    db.commit()
    db.refresh(notification)
    
    return notification_to_camel(notification)


@router.patch("/mark-all-read")
async def mark_all_notifications_read(db: Session = Depends(get_db)):
    """Mark all notifications as read."""
    db.query(Notification).filter(Notification.read == False).update({"read": True})
    db.commit()
    
    return MessageResponse(message="All notifications marked as read")


@router.websocket("/ws")
async def notifications_ws(websocket: WebSocket):
    """WebSocket endpoint for streaming notifications to the frontend."""
    await notification_manager.connect(websocket)
    try:
        # Keep the connection alive; backend pushes events as they occur.
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        notification_manager.disconnect(websocket)


@router.delete("/{notification_id}", response_model=MessageResponse)
async def delete_notification(notification_id: str, db: Session = Depends(get_db)):
    """Delete a notification."""
    notification = db.query(Notification).filter(Notification.id == notification_id).first()
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Notification with id {notification_id} not found"
        )
    
    db.delete(notification)
    db.commit()
    
    return MessageResponse(message=f"Notification {notification_id} deleted successfully")
