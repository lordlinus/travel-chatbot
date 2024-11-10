import asyncio
import datetime
import random
from typing import List

from autogen_core.base import MessageContext
from autogen_core.components import (DefaultTopicId, RoutedAgent,
                                     message_handler, type_subscription)
from autogen_core.components.models import LLMMessage, SystemMessage
from autogen_core.components.tools import FunctionTool, Tool
from typing_extensions import Annotated

from ..data_types import (AgentStructuredResponse, CarRental, EndUserMessage,
                          GroupChatMessage, HandoffMessage, TravelRequest)
from ..otlp_tracing import logger


async def simulate_car_rental_booking(
    rental_city: Annotated[str, "The city where the car rental will take place."],
    rental_start_date: Annotated[
        str, "The start date of the car rental in the format 'YYYY-MM-DD'."
    ],
    rental_end_date: Annotated[
        str, "The end date of the car rental in the format 'YYYY-MM-DD'."
    ],
) -> CarRental:
    car_options = [
        {"car_type": "Sedan", "company": "Avis", "price_per_day": 50},
        {"car_type": "SUV", "company": "Hertz", "price_per_day": 80},
        {"car_type": "Convertible", "company": "Budget", "price_per_day": 100},
        {"car_type": "Minivan", "company": "Enterprise", "price_per_day": 70},
        {"car_type": "Compact", "company": "Thrifty", "price_per_day": 40},
        {"car_type": "Luxury", "company": "Alamo", "price_per_day": 150},
        {"car_type": "Pickup Truck", "company": "National", "price_per_day": 90},
        {"car_type": "Electric", "company": "Tesla Rentals", "price_per_day": 120},
        {"car_type": "Hybrid", "company": "Green Wheels", "price_per_day": 60},
        {"car_type": "Sports Car", "company": "Exotic Rentals", "price_per_day": 200},
    ]

    selected_car = random.choice(car_options)
    start_date = datetime.datetime.strptime(rental_start_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(rental_end_date, "%Y-%m-%d")
    rental_days = (end_date - start_date).days
    total_price = rental_days * selected_car["price_per_day"]
    booking_reference = f"CR-{random.randint(1000, 9999)}-{rental_city[:3].upper()}"

    car_rental_details = CarRental(
        rental_city=rental_city,
        rental_start_date=rental_start_date,
        rental_end_date=rental_end_date,
        car_type=selected_car["car_type"],
        company=selected_car["company"],
        total_price=total_price,
        booking_reference=booking_reference,
    )

    await asyncio.sleep(2)
    return car_rental_details


def get_car_rental_tool() -> List[Tool]:
    return [
        FunctionTool(
            name="simulate_car_rental_booking",
            func=simulate_car_rental_booking,
            description="Simulates a car rental booking process based on user preferences.",
        )
    ]


@type_subscription("car_rental")
class CarRentalAgent(RoutedAgent):
    def __init__(self) -> None:
        super().__init__("CarRentalAgent")
        logger.info("CarRentalAgent initialized")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                "You are a helpful AI assistant that can advise on car rental bookings based on user preferences."
            )
        ]

    @message_handler
    async def handle_message(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        logger.info(f"CarRentalAgent received message: {message.content}")
        if "travel plan" in message.content.lower():
            await self.publish_message(
                HandoffMessage(content=message.content, source=self.id.type),
                DefaultTopicId(type="router", source=ctx.topic_id.source),
            )
            return

        # You would typically call a LLM to extract the requirement or have a function call here
        requirements = {
            "rental_city": (
                "New York" if "new york" in message.content.lower() else "Unknown"
            ),
            "rental_start_date": "2023-12-21",
            "rental_end_date": "2023-12-26",
        }
        response = await simulate_car_rental_booking(
            requirements["rental_city"],
            requirements["rental_start_date"],
            requirements["rental_end_date"],
        )
        await self.publish_message(
            AgentStructuredResponse(
                agent_type=self.id.type,
                data=response,
                message=f"Car rented: {response}",
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )

    @message_handler
    async def handle_travel_request(
        self, message: TravelRequest, ctx: MessageContext
    ) -> GroupChatMessage:
        logger.info(
            f"CarRentalAgent received travel request: TravelRequest - {message.content}"
        )
        requirements = {
            "rental_city": (
                "New York" if "new york" in message.content.lower() else "Unknown"
            ),
            "rental_start_date": "2023-12-21",
            "rental_end_date": "2023-12-26",
        }
        response = await simulate_car_rental_booking(
            requirements["rental_city"],
            requirements["rental_start_date"],
            requirements["rental_end_date"],
        )
        return GroupChatMessage(
            source=self.id.type,
            content=f"Car rented: {response}",

        )
