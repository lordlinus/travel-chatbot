import json
from typing import List

from autogen_core.base import MessageContext
from autogen_core.components import (DefaultTopicId, RoutedAgent,
                                     message_handler, type_subscription)
from autogen_core.components.models import (LLMMessage, SystemMessage,
                                            UserMessage)
from autogen_ext.models import AzureOpenAIChatCompletionClient

from ..data_types import (AgentStructuredResponse, DestinationInfo,
                          EndUserMessage, GroupChatMessage, TravelRequest)
from ..otlp_tracing import logger


# Destination Agent
@type_subscription(topic_type="destination_info")
class DestinationAgent(RoutedAgent):
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
    ) -> None:
        super().__init__("DestinationAgent")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                "You are a helpful AI assistant that helps with destination information."
            )
        ]
        self._model_client = model_client

    @message_handler
    async def handle_message(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        logger.info(
            f"DestinationAgent received travel request: EndUserMessage {message.content}"
        )
        # Provide destination information
        try:
            response_content = await self._model_client.create(
                [
                    UserMessage(
                        content=f"Provide info for {message.content}",
                        source="DestinationAgent",
                    )
                ],
                extra_create_args={"response_format": DestinationInfo},
            )
            destination_info_structured = DestinationInfo.model_validate(
                json.loads(response_content.content)
            )
        except Exception as e:
            logger.error(f"Failed to parse destination response: {str(e)}")
            destination_info_structured = DestinationInfo()

        await self.publish_message(
            AgentStructuredResponse(
                agent_type=self.id.type,
                data=destination_info_structured,
                message=message.content,
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )

    @message_handler
    async def handle_travel_request(
        self, message: TravelRequest, ctx: MessageContext
    ) -> GroupChatMessage:
        logger.info(
            f"DestinationAgent received travel request: TravelRequest: {message.content}"
        )
        # Provide destination information
        try:
            response_content = await self._model_client.create(
                [
                    UserMessage(
                        content=f"You have been given a subtask: {message.content} as part of a travel plan. The initial task is {message.original_task}",
                        source="DestinationAgent",
                    )
                ],
                extra_create_args={"response_format": DestinationInfo},
            )
            destination_info_structured = DestinationInfo.model_validate(
                json.loads(response_content.content)
            )
        except Exception as e:
            logger.error(f"Failed to parse destination response: {str(e)}")
            destination_info_structured = DestinationInfo()

        return GroupChatMessage(
            source=self.id.type,
            content=destination_info_structured.model_dump_json(),
        )
