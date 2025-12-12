/**
 * WebSocket Service for real-time notifications
 */

export interface ScannerNotification {
  type: 'scanner_notification';
  data: {
    patient_id: string;
    patient_name: string;
    classification: string;
    confidence: number;
    severity: 'critical' | 'warning' | 'info' | 'success';
    message: string;
    findings?: string[];
    recommendation?: string;
    notification_subtype?: 'abnormality_detected' | 'analysis_complete';
    timestamp: string;
  };
}

export interface ConnectionStatus {
  type: 'connection_status';
  data: {
    status: 'connected' | 'disconnected' | 'reconnecting';
    timestamp: string;
  };
}

export type WebSocketMessage = ScannerNotification | ConnectionStatus;

type MessageHandler = (message: WebSocketMessage) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 3000; // 3 seconds
  private messageHandlers: Set<MessageHandler> = new Set();
  private isIntentionallyClosed = false;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;

  /**
   * Connect to the WebSocket server
   */
  connect(url?: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      console.log('[WS] Already connected');
      return;
    }

    this.isIntentionallyClosed = false;
    const wsUrl = url || this.getDefaultUrl();
    
    console.log(`[WS] Connecting to ${wsUrl}...`);
    
    try {
      this.ws = new WebSocket(wsUrl);
      this.setupEventListeners();
    } catch (error) {
      console.error('[WS] Failed to create WebSocket:', error);
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the WebSocket server
   */
  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.clearReconnectTimer();
    
    if (this.ws) {
      console.log('[WS] Disconnecting...');
      this.ws.close(1000, 'Client disconnected');
      this.ws = null;
    }
    
    this.reconnectAttempts = 0;
  }

  /**
   * Subscribe to WebSocket messages
   */
  subscribe(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    
    // Return unsubscribe function
    return () => {
      this.messageHandlers.delete(handler);
    };
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Get connection state
   */
  getState(): 'connecting' | 'open' | 'closing' | 'closed' | 'disconnected' {
    if (!this.ws) return 'disconnected';
    
    switch (this.ws.readyState) {
      case WebSocket.CONNECTING:
        return 'connecting';
      case WebSocket.OPEN:
        return 'open';
      case WebSocket.CLOSING:
        return 'closing';
      case WebSocket.CLOSED:
        return 'closed';
      default:
        return 'disconnected';
    }
  }

  private getDefaultUrl(): string {
    // Determine WebSocket URL based on current location
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = import.meta.env.VITE_API_HOST || 'localhost:8000';
    return `${protocol}//${host}/ws/notifications`;
  }

  private setupEventListeners(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('[WS] Connected successfully');
      this.reconnectAttempts = 0;
      
      this.notifyHandlers({
        type: 'connection_status',
        data: {
          status: 'connected',
          timestamp: new Date().toISOString(),
        },
      });
    };

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WebSocketMessage;
        console.log('[WS] Received message:', message);
        this.notifyHandlers(message);
      } catch (error) {
        console.error('[WS] Failed to parse message:', error, event.data);
      }
    };

    this.ws.onerror = (event) => {
      console.error('[WS] WebSocket error:', event);
    };

    this.ws.onclose = (event) => {
      console.log(`[WS] Connection closed: code=${event.code}, reason=${event.reason}`);
      
      this.notifyHandlers({
        type: 'connection_status',
        data: {
          status: 'disconnected',
          timestamp: new Date().toISOString(),
        },
      });

      if (!this.isIntentionallyClosed) {
        this.scheduleReconnect();
      }
    };
  }

  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.log('[WS] Max reconnection attempts reached');
      return;
    }

    this.clearReconnectTimer();
    this.reconnectAttempts++;
    
    const delay = this.reconnectDelay * Math.pow(1.5, this.reconnectAttempts - 1);
    console.log(`[WS] Scheduling reconnect attempt ${this.reconnectAttempts} in ${delay}ms`);

    this.notifyHandlers({
      type: 'connection_status',
      data: {
        status: 'reconnecting',
        timestamp: new Date().toISOString(),
      },
    });

    this.reconnectTimer = setTimeout(() => {
      this.connect();
    }, delay);
  }

  private clearReconnectTimer(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
  }

  private notifyHandlers(message: WebSocketMessage): void {
    this.messageHandlers.forEach((handler) => {
      try {
        handler(message);
      } catch (error) {
        console.error('[WS] Handler error:', error);
      }
    });
  }
}

// Singleton instance
export const websocketService = new WebSocketService();
