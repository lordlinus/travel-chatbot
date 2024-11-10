import json
from collections import deque

from autogen_core.base import MessageContext
from autogen_core.components import (DefaultTopicId, RoutedAgent,
                                     message_handler, type_subscription)
from autogen_core.components.models import SystemMessage
from autogen_ext.models import AzureOpenAIChatCompletionClient

from ..data_types import (AgentStructuredResponse, EndUserMessage, Greeter,
                          HandoffMessage, TravelPlan)
from ..otlp_tracing import logger
from ..registry import AgentRegistry
from ..session_state import SessionStateManager

agent_registry = AgentRegistry()


@type_subscription(topic_type="router")
class SemanticRouterAgent(RoutedAgent):
    """
    The SemanticRouterAgent routes incoming messages to appropriate agents based on the intent.

    Attributes:
        name (str): Name of the agent.
        model_client (AzureOpenAIChatCompletionClient): The model client for agent routing.
        agent_registry (AgentRegistry): The registry containing agent information.
        intent_classifier (IntentClassifier): Classifier to determine intent from user messages.
        session_manager (SessionStateManager): Manages the session state for each user.
    """

    def __init__(
        self,
        name: str,
        model_client: AzureOpenAIChatCompletionClient,
        agent_registry: AgentRegistry,
        session_manager: SessionStateManager,
    ) -> None:
        super().__init__("SemanticRouterAgent")
        self._name = name
        self._model_client = model_client
        self._registry = agent_registry
        self._session_manager = session_manager

    @message_handler
    async def route_message(self, message: EndUserMessage, ctx: MessageContext) -> None:
        """
        Routes user messages to appropriate agents based on conversation context.

        Args:
            message (EndUserMessage): The incoming user message.
            ctx (MessageContext): Context information for the message.
        """
        session_id = ctx.topic_id.source

        # Add the current message to session history
        self._session_manager.add_to_history(session_id, message)

        # Analyze conversation history for better context
        history = self._session_manager.get_history(session_id)
        logger.info("Analyzing conversation history for context")

        travel_plan: TravelPlan = await self._get_agents_to_route(message, history)
        logger.info(f"Routing message to agents: {travel_plan}")

        if travel_plan.is_greeting:
            logger.info("User greeting detected")
            await self.publish_message(
                AgentStructuredResponse(
                    agent_type="default_agent",
                    data=Greeter(
                        greeting="Greetings, Adventurer! 🌍 Ready to embark on your next journey? I'm here to turn your travel dreams into reality. Let's dive into the details and craft an unforgettable adventure together. From flights to sights, I've got you covered. Let's get started!"
                    ),
                    message=f"User greeting detected: {message.content}",
                ),
                DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
            )
            return

        if not travel_plan.subtasks:
            logger.info("No agents selected for routing")
            return

        # If only one agent is involved, send the message directly
        if len(travel_plan.subtasks) == 1:
            assigned_agent = travel_plan.subtasks[0].assigned_agent
            logger.info(f"Routing message to agent: {assigned_agent}")
            await self.publish_message(
                EndUserMessage(content=message.content, source=message.source),
                DefaultTopicId(type=assigned_agent, source=session_id),
            )
        else:
            # If more than one agent is involved, send the message to GroupChatManager for coordination
            logger.info(
                f"Routing message to GroupChatManager for coordination: {[subtask.assigned_agent for subtask in travel_plan.subtasks]}"
            )
            await self.publish_message(
                travel_plan,
                DefaultTopicId(type="group_chat_manager", source=session_id),
            )

    @message_handler
    async def handle_handoff(
        self, message: HandoffMessage, ctx: MessageContext
    ) -> None:
        """
        Handles handoff messages from other agents.

        Args:
            message (HandoffMessage): The handoff message from another agent.
            ctx (MessageContext): Context information for the message.
        """
        session_id = ctx.topic_id.source
        logger.info(f"Received handoff message from {message.source}")

        # Clear session if conversation is complete, otherwise continue routing
        if message.original_task and "complete" in message.content.lower():
            self._session_manager.clear_session(session_id)
        else:
            await self.route_message(
                EndUserMessage(content=message.content, source=message.source), ctx
            )

    async def _get_agents_to_route(
        self, message: EndUserMessage, history: deque
    ) -> TravelPlan:
        """
        Determines the appropriate agents to route the message to based on context.

        Args:
            message (EndUserMessage): The incoming user message.
            history (deque): The history of messages in the session.

        Returns:
            TravelPlan: A travel plan indicating which agents should handle the subtasks.
        """
        # System prompt to determine the appropriate agents to handle the message
        logger.info(f"Analyzing message: {message.content}")
        try:
            logger.info(
                f"Getting planner prompt for message: {message.content} and history: {[msg.content for msg in history]}"
            )
            system_message = agent_registry.get_planner_prompt(
                message=message, history=history
            )
            # logger.info(f"System message: {system_message}")
        except Exception as e:
            logger.error(e)

        try:
            response = await self._model_client.create(
                [SystemMessage(system_message)],
                extra_create_args={"response_format": TravelPlan},
            )
            my_travel_plan: TravelPlan = TravelPlan.model_validate(
                json.loads(response.content)
            )
            if my_travel_plan.is_greeting:
                logger.info("User greeting detected")
                my_travel_plan.subtasks = [
                    {
                        "task_details": f"Greeting - {message.content}",
                        "assigned_agent": "default_agent",
                    }
                ]

            logger.info(f"Received travel plan: {my_travel_plan}")
            return my_travel_plan
        except Exception as e:
            logger.error(f"Failed to parse activities response: {str(e)}")
            return TravelPlan(subtasks=[])
