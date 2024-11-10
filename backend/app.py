import sys
import os

# Add the root directory of the project to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Union

from autogen_core.base import MessageContext
from autogen_core.components import (
    DefaultTopicId,
    RoutedAgent,
    default_subscription,
    message_handler,
)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.websockets import WebSocketState

from backend.data_types import AgentResponse, EndUserMessage, AgentStructuredResponse
from backend.otlp_tracing import logger
from backend.utils import initialize_agent_runtime, get_web_pub_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles the startup and shutdown lifespan events for the FastAPI application.

    Initializes the agent runtime and registers the UserProxyAgent.
    """
    global agent_runtime
    global user_proxy_agent_instance
    global web_pubsub_client
    # Initialize the agent runtime
    agent_runtime = await initialize_agent_runtime()

    # Create a WebPubSubServiceClient
    web_pubsub_client = get_web_pub_client()

    # Register the UserProxyAgent instance with the AgentRuntime
    await UserProxyAgent.register(agent_runtime, "user_proxy", lambda: UserProxyAgent())

    yield  # This separates the startup and shutdown logic

    # Cleanup logic goes here
    agent_runtime = None
    user_proxy_agent_instance = None


# Define FastAPI app
app = FastAPI(lifespan=lifespan)

# Instrument FastAPI app for tracing
FastAPIInstrumentor.instrument_app(app)

# Initialize the agent runtime at startup
agent_runtime = None
user_proxy_agent_instance = None  # Global variable to store the UserProxyAgent instance


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for user sessions.
    """

    def __init__(self):
        self.connections: Dict[str, WebSocket] = {}

    def add_connection(self, session_id: str, websocket: WebSocket) -> None:
        """
        Adds a new WebSocket connection to the manager.

        Args:
            session_id (str): The unique identifier for the session.
            websocket (WebSocket): The WebSocket connection.
        """
        self.connections[session_id] = websocket

    def remove_connection(self, session_id: str) -> None:
        """
        Removes a WebSocket connection from the manager.

        Args:
            session_id (str): The unique identifier for the session.
        """
        if session_id in self.connections:
            del self.connections[session_id]

    async def handle_websocket(self, websocket: WebSocket, session_id: str):
        """
        Handles incoming WebSocket messages and manages connection lifecycle.

        Args:
            websocket (WebSocket): The WebSocket connection.
            session_id (str): The unique identifier for the session.
        """
        await websocket.accept()
        self.add_connection(session_id, websocket)
        try:
            while True:
                user_message_text = await websocket.receive_text()
                chat_id = str(uuid.uuid4())
                user_message = EndUserMessage(content=user_message_text, source="User")

                logger.info(f"Received message with chat_id: {chat_id}")

                # Publish the user's message to the agent
                await agent_runtime.publish_message(
                    user_message,
                    DefaultTopicId(type="user_proxy", source=session_id),
                )
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            logger.info(f"WebSocket connection closed: {session_id}")
        except Exception as e:
            logger.error(f"Exception in WebSocket connection {session_id}: {str(e)}")
        finally:
            self.remove_connection(session_id)
            try:
                if websocket.client_state != WebSocketState.DISCONNECTED:
                    await websocket.close()
            except WebSocketDisconnect:
                logger.info(f"WebSocket already closed: {session_id}")


# User Proxy Agent
@default_subscription
class UserProxyAgent(RoutedAgent):
    """
    Acts as a proxy between the user and the routing agent.
    """

    def __init__(self) -> None:
        super().__init__("UserProxyAgent")

    @message_handler
    async def handle_agent_response(
        self,
        message: AgentStructuredResponse,
        ctx: MessageContext,
    ) -> None:
        """
        Sends the agent's response back to the user via WebSocket.

        Args:
            message (AgentStructuredResponse): The agent's response message.
            ctx (MessageContext): The message context.
        """
        logger.info(f"UserProxyAgent received agent response: {message}")
        session_id = ctx.topic_id.source
        try:
            websocket = connection_manager.connections.get(session_id)
            if websocket:
                await websocket.send_text(f"{message.model_dump_json()}")
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {str(e)}")

    @message_handler
    async def handle_user_message(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        """
        Forwards the user's message to the router for further processing.

        Args:
            message (EndUserMessage): The user's message.
            ctx (MessageContext): The message context.
        """
        logger.info(f"UserProxyAgent received user message: {message.content}")
        # Forward the message to the router
        await self.publish_message(
            EndUserMessage(content=message.content, source=message.source),
            DefaultTopicId(type="router", source=ctx.topic_id.source),
        )


# Initialize the WebSocket connection manager
connection_manager = WebSocketConnectionManager()


# WebSocket endpoint to handle user messages
@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for handling user chat messages.

    Args:
        websocket (WebSocket): The WebSocket connection.
    """
    session_id = str(uuid.uuid4())
    await connection_manager.handle_websocket(websocket, session_id)


@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify that the service is running.
    """
    return {"status": "ok"}


# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.app:app", host="127.0.0.1", port=8000, reload=True)
