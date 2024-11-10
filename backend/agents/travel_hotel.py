import datetime
import random
from typing import Dict, List

from autogen_core.base import AgentId, MessageContext
from autogen_core.components import (
    DefaultTopicId,
    RoutedAgent,
    message_handler,
    type_subscription,
)
from autogen_core.components.models import LLMMessage, SystemMessage, UserMessage
from autogen_core.components.tool_agent import tool_agent_caller_loop
from autogen_core.components.tools import FunctionTool, Tool
from autogen_ext.models import AzureOpenAIChatCompletionClient
from typing_extensions import Annotated

from ..data_types import (
    AgentStructuredResponse,
    EndUserMessage,
    GroupChatMessage,
    HandoffMessage,
    TravelRequest,
    HotelBooking,
)
from ..otlp_tracing import logger


async def create_hotel_booking(
    city: Annotated[str, "The city where the hotel booking will take place."],
    check_in_date: Annotated[
        str, "The check-in date of the hotel booking in the format 'YYYY-MM-DD'."
    ],
    check_out_date: Annotated[
        str, "The check-out date of the hotel booking in the format 'YYYY-MM-DD'."
    ],
) -> HotelBooking:
    # Simulate available hotel options
    logger.info(
        f"Function call: Creating hotel booking for {city} from {check_in_date} to {check_out_date}"
    )
    hotel_options = [
        {"hotel_name": "Hilton", "room_type": "Deluxe", "price_per_night": 200},
        {"hotel_name": "Marriott", "room_type": "Standard", "price_per_night": 150},
        {"hotel_name": "Hyatt", "room_type": "Suite", "price_per_night": 300},
        {"hotel_name": "Sheraton", "room_type": "Executive", "price_per_night": 250},
        {"hotel_name": "Holiday Inn", "room_type": "Standard", "price_per_night": 100},
    ]

    # Randomly select a hotel option
    selected_hotel = random.choice(hotel_options)

    # Calculate the number of nights
    check_in = datetime.datetime.strptime(check_in_date, "%Y-%m-%d")
    check_out = datetime.datetime.strptime(check_out_date, "%Y-%m-%d")
    num_nights = (check_out - check_in).days

    # Calculate total price for the stay
    total_price = num_nights * selected_hotel["price_per_night"]

    # Create a booking reference number
    booking_reference = f"HT-{random.randint(1000, 9999)}-{city[:3].upper()}"

    hotel_booking_details = HotelBooking(
        city=city,
        check_in_date=check_in_date,
        check_out_date=check_out_date,
        hotel_name=selected_hotel["hotel_name"],
        room_type=selected_hotel["room_type"],
        total_price=total_price,
        booking_reference=booking_reference,
    )

    logger.info(f"Hotel booking details: {hotel_booking_details}")

    return hotel_booking_details


def get_hotel_booking_tool() -> List[Tool]:
    return [
        FunctionTool(
            create_hotel_booking,
            description="This function performs the process of booking a hotel. It takes the city, check-in date, and check-out date as input and returns the hotel booking details, including the city, check-in and check-out dates, hotel name, room type, total price, and booking reference. This function is useful when the user wants to book a hotel",
        )
    ]


# Hotel Agent with Handoff Logic
@type_subscription("hotel_booking")
class HotelAgent(RoutedAgent):
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
        tools: List[Tool],
        tool_agent_type: str,
    ) -> None:
        super().__init__("HotelAgent")
        self._system_messages: List[LLMMessage] = [
            SystemMessage("You are a helpful AI assistant that can make hotel booking.")
        ]
        self._model_client = model_client
        self._tools = tools
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)

    async def _process_request(self, message_content: str, ctx: MessageContext) -> str:
        # Create a session for the activities agent
        session: List[LLMMessage] = [
            UserMessage(content=message_content, source="user")
        ]
        # Run the caller loop
        try:
            messages = await tool_agent_caller_loop(
                caller=self,
                tool_agent_id=self._tool_agent_id,
                model_client=self._model_client,
                input_messages=session,
                tool_schema=self._tools,
                cancellation_token=ctx.cancellation_token,
            )
            logger.info(f"Tool agent caller loop completed: {messages}")
        except Exception as e:
            logger.error(f"Tool agent caller loop failed: {str(e)}")
            return "Failed to book hotel. Please try again."

        # Ensure the final message content is a string
        assert isinstance(messages[-1].content, str)
        return messages[-1].content

    @message_handler
    async def handle_message(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        logger.info(f"HotelAgent received message - EndUserMessage: {message.content}")
        if "travel plan" in message.content.lower():
            # Cannot handle complex travel plans, hand off back to router
            await self.publish_message(
                HandoffMessage(content=message.content, source=self.id.type),
                DefaultTopicId(type="router", source=ctx.topic_id.source),
            )
            return

        response_content = await self._process_request(message.content, ctx)
        simulated_func_call = await create_hotel_booking(
            city="Singapore",
            check_in_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            check_out_date=(
                datetime.datetime.now() + datetime.timedelta(days=5)
            ).strftime("%Y-%m-%d"),
        )
        # Publish the response to the user proxy
        await self.publish_message(
            AgentStructuredResponse(
                agent_type=self.id.type,
                data=simulated_func_call,
                message=f"{response_content}",
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )

    @message_handler
    async def handle_travel_request(
        self, message: TravelRequest, ctx: MessageContext
    ) -> GroupChatMessage:
        logger.info(
            f"HotelAgent received travel request - TravelRequest: {message.content}"
        )
        response_content = await self._process_request(message.content, ctx)
        logger.info(f"HotelAgent response: {response_content}")

        simulated_func_call = await create_hotel_booking(
            city="Singapore",
            check_in_date=datetime.datetime.now().strftime("%Y-%m-%d"),
            check_out_date=(
                datetime.datetime.now() + datetime.timedelta(days=5)
            ).strftime("%Y-%m-%d"),
        )

        return GroupChatMessage(
            source=self.id.type,
            content=f"{response_content}",
        )
