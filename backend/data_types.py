from datetime import date
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel

# Improvements and suggestions:
# 1. Add better documentation to each model for clarity and easy onboarding.
# 2. Introduce a BaseAgentMessage for standardizing agent messages.
# 3. Refactor repetitive fields into common base classes where applicable.
# 4. Make use of `Union` types to create more generic message classes where possible.
# 5. Consolidate similar message classes to reduce redundancy.


# Base class for messages exchanged between agents and users
class BaseAgentMessage(BaseModel):
    source: str
    timestamp: Optional[date] = None


# Unified User and Agent Message Base Class
class EndUserMessage(BaseAgentMessage):
    content: str


class AgentResponse(BaseAgentMessage):
    content: str


class GroupChatMessage(BaseAgentMessage):
    """
    Represents a message exchanged during a group chat session.

    Attributes:
        content (str): The content of the group chat message.
        group_id (str): Identifier for the group chat session.
        sender (str): The identifier of the sender.
        recipients (Optional[List[str]]): Optional list of recipients for the message.
        message_type (Optional[str]): Type of message, such as 'user', 'system', or 'response'.
    """

    content: str
    group_id: Optional[str] = ""
    sender: Optional[str] = ""
    recipients: Optional[List[str]] = None
    message_type: Optional[str] = "user"


class GroupChatResponse(BaseAgentMessage):
    content: str
    message_type: Optional[str] = "response"


class RequestToSpeak(BaseAgentMessage):
    pass


class TravelRequest(BaseAgentMessage):
    content: str
    original_task: Optional[str] = None


class HandoffMessage(BaseAgentMessage):
    content: str


# Activities Information Data Model
class ActivitiesDetail(BaseModel):
    activity_name: str
    activity_type: str
    activity_description: str


class Activities(BaseModel):
    destination_city: str
    activities: List[ActivitiesDetail]


class Greeter(BaseModel):
    greeting: str


# Destination Information Data Model
class DestinationInfo(BaseModel):
    city: str
    country: str
    description: str
    best_time_to_visit: str
    average_temperature: str
    currency: str
    language: str
    similar_destinations: List[str]


# Flight Booking Data Model
class FlightBooking(BaseModel):
    departure_city: str
    destination_city: str
    departure_date: date
    return_date: date
    airline: str
    flight_number: str
    total_price: float
    booking_reference: str
    number_of_passengers: int


# Hotel Booking Data Model
class HotelBooking(BaseModel):
    city: str
    check_in_date: str
    check_out_date: str
    hotel_name: str
    room_type: str
    total_price: float
    booking_reference: str


# Car Rental Data Model
class CarRental(BaseModel):
    rental_city: str
    rental_start_date: str
    rental_end_date: str
    car_type: str
    company: str
    total_price: float
    booking_reference: str


# Enum to Define Agent Types
class AgentEnum(str, Enum):
    FlightBooking = "flight_booking"
    HotelBooking = "hotel_booking"
    CarRental = "car_rental"
    ActivitiesBooking = "activities_booking"
    DestinationInfo = "destination_info"
    DefaultAgent = "default_agent"
    GroupChatManager = "group_chat_manager"


# Travel SubTask Model
class TravelSubTask(BaseModel):
    task_details: str
    assigned_agent: AgentEnum

    class Config:
        use_enum_values = True  # To serialize enums as their values


class TravelPlan(BaseModel):
    main_task: str
    subtasks: List[TravelSubTask]
    is_greeting: bool


# Generic Response Wrapper
class AgentStructuredResponse(BaseModel):
    agent_type: AgentEnum
    data: Union[
        Activities,
        DestinationInfo,
        FlightBooking,
        HotelBooking,
        CarRental,
        Greeter,
        GroupChatMessage,
    ]
    message: Optional[str] = None  # Additional message or notes from the agent


# Resource Node Model
class Resource(BaseModel):
    """
    Represents a resource node retrieved during chat interactions.

    Attributes:
        content (str): The textual content of the resource.
        node_id (str): The identifier of the node.
        score (Optional[float]): Score representing the relevance of the resource.
    """

    content: str
    node_id: str
    score: Optional[float] = None
