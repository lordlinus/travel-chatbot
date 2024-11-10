import random
from typing import List

from autogen_core.base import MessageContext
from autogen_core.components import (DefaultTopicId, RoutedAgent,
                                     message_handler, type_subscription)
from autogen_core.components.tools import FunctionTool, Tool

from ..data_types import (AgentStructuredResponse,
                          EndUserMessage, FlightBooking, GroupChatMessage,
                          HandoffMessage, TravelRequest)
from ..otlp_tracing import logger


async def simulate_flight_booking(
    departure_city: str = "New York",
    destination_city: str = "Paris",
    departure_date: str = "2023-12-20",
    return_date: str = "2023-12-30",
    number_of_passengers: int = 2,
) -> FlightBooking:
    flight_options = [
        {"airline": "Air France", "flight_number": "AF123", "price_per_ticket": 200},
        {"airline": "Delta", "flight_number": "DL456", "price_per_ticket": 250},
        {
            "airline": "British Airways",
            "flight_number": "BA789",
            "price_per_ticket": 300,
        },
        {"airline": "Lufthansa", "flight_number": "LH101", "price_per_ticket": 220},
        {"airline": "Emirates", "flight_number": "EK202", "price_per_ticket": 400},
    ]

    selected_flight = random.choice(flight_options)
    total_price = 2 * selected_flight["price_per_ticket"]
    booking_reference = (
        f"FL-{random.randint(1000, 9999)}-{destination_city[:3].upper()}"
    )

    return FlightBooking(
        departure_city=departure_city,
        destination_city=destination_city,
        departure_date=departure_date,
        return_date=return_date,
        airline=selected_flight["airline"],
        flight_number=selected_flight["flight_number"],
        total_price=total_price,
        booking_reference=booking_reference,
        number_of_passengers=number_of_passengers,
    )


def get_flight_booking_tool() -> List[Tool]:
    return [
        FunctionTool(
            name="simulate_flight_booking",
            func=simulate_flight_booking,
            description="This function simulates the process of booking a flight. It takes the departure city, destination city, departure date, and return date as input and returns the flight booking details, including the departure and destination cities, departure and return dates, airline, flight number, total price, and booking reference. This function is useful when the user wants to book a flight.",
        )
    ]


# This agent is responsible for handling flight booking requests from the user.
# It simulates a flight booking process and returns a response with the booking details.
# NOTE: There is no LLM in this agent and we are simulating the flight booking process.
@type_subscription(topic_type="flight_booking")
class FlightAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("FlightAgent")

    @message_handler
    async def handle_message(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        logger.info(f"FlightAgent received message: {message.content}")
        if "travel plan" in message.content.lower():
            await self.publish_message(
                HandoffMessage(content=message.content, source=self.id.type),
                DefaultTopicId(type="router", source=ctx.topic_id.source),
            )
            return

        response = await simulate_flight_booking()
        await self.publish_message(
            AgentStructuredResponse(
                agent_type=self.id.type,
                data=response,
                message=f"Simulated response: Flight booking processed successfully for query - {message.content}",
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )

    @message_handler
    async def handle_travel_request(
        self, message: TravelRequest, ctx: MessageContext
    ) -> GroupChatMessage:
        logger.info(f"FlightAgent received travel request sub-task: {message.content}")

        response = await simulate_flight_booking()
        return GroupChatMessage(
            source=self.id.type,
            content=f"Flight booking processed: {response}",
        )
