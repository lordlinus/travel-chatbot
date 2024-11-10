from typing import Optional, List, Dict, Any
from .otlp_tracing import logger
from backend.agents.travel_flight import get_flight_booking_tool
from backend.agents.travel_hotel import get_hotel_booking_tool
from backend.agents.travel_car import get_car_rental_tool
from backend.agents.travel_activities import get_travel_activity_tools

from backend.data_types import EndUserMessage


class AgentRegistry:
    def __init__(self):
        self.agents = {
            "default_agent": {
                "agent_type": "default_agent",
                "description": "Handles user greetings, salutations, and general travel-related queries that do not fit into other specific categories. Route messages here if they are greetings (e.g., 'hi', 'hello', 'good morning') or general travel queries that do not specify a destination or service.",
                "examples": "'Hello', 'Hi there!', 'Good morning', 'I want to plan a trip.'",
            },
            "destination_info": {
                "agent_type": "destination_info",
                "description": "Provides detailed information about a specified destination city. Use this agent when the user requests information about a destination city by name (e.g., 'Tell me about Paris', 'What can I do in Tokyo?').",
                "examples": "'Tell me about London', 'What's the weather like in Paris?', 'Top attractions in New York City?'",
            },
            "flight_booking": {
                "agent_type": "flight_booking",
                "description": "Helps in providing flight information. Assign messages here when the user wants to book or inquire about flights.",
                "examples": "'Book me a flight to London', 'I need flight options from LA to NYC.'",
            },
            "hotel_booking": {
                "agent_type": "hotel_booking",
                "description": "Helps in booking hotels. Direct messages here if the user wants to book or ask about hotels.",
                "examples": "'Find me a hotel in Berlin', 'I need hotel reservations for next weekend.'",
            },
            "car_rental": {
                "agent_type": "car_rental",
                "description": "Helps in booking car rentals. Use this agent for car rental bookings or inquiries.",
                "examples": "'I need to rent a car in Miami', 'Car rental options in San Francisco?'",
            },
            "activities_booking": {
                "agent_type": "activities_booking",
                "description": "Helps in providing activities information. Route messages here when the user is interested in booking activities or seeking information about activities in a specific location.",
                "examples": "'What events are happening in Chicago?', 'Book me a tour in Rome.'",
            },
        }

        self.agent_tools = self.retrieve_all_agent_tools()

    def retrieve_all_agent_tools(self) -> List[Dict[str, Any]]:
        tools = []
        agent_tools = {
            "flight_booking": get_flight_booking_tool(),
            "hotel_booking": get_hotel_booking_tool(),
            "car_rental": get_car_rental_tool(),
            "activities_booking": get_travel_activity_tools(),
        }

        for agent, tool_list in agent_tools.items():
            for tool in tool_list:
                tools.append(
                    {
                        "agent": agent,
                        "function": tool.name,
                        "description": tool.description,
                        "arguments": list(tool.schema["parameters"]["properties"]),
                    }
                )
        # logger.info(f"Agent tools: {tools}")

        return tools

    async def get_agent(self, intent: str) -> Optional[dict]:
        logger.info(f"AgentRegistry: Getting agent for intent: {intent}")
        return self.agents.get(intent)

    def get_planner_prompt(self, message: EndUserMessage, history) -> str:
        agent_details = {}
        for agent in self.agents.values():
            agent_details[agent["agent_type"]] = {
                "description": agent["description"],
                "examples": agent.get("examples", ""),
                "functions": [],
            }

        for tool in self.agent_tools:
            agent = tool["agent"]
            function = tool["function"]
            arguments = tool["arguments"]
            agent_details[agent]["functions"].append(
                {"function": function, "arguments": arguments}
            )

        agent_descriptions = "\n".join(
            f"""
    Agent Type: {agent_type}
    Description: {details['description']}
    Examples: {details['examples']}
    {chr(10).join([f"- Function: {func['function']} (Arguments: {', '.join(func['arguments'])})" for func in details['functions']]) if details['functions'] else ""}
    """
            for agent_type, details in agent_details.items()
        )

        # logger.info(f"Agent descriptions: {agent_descriptions}")

        planner_prompt = """
    You are an orchestration agent.
    Your job is to decide which agents to run based on the user's request and the conversation history.
    Below are the available agents:

    {agent_descriptions}

    The current user message: {message}
    Conversation history so far: {history}

    Your response should only include the selected agent and a brief justification for your choice, without any additional text.
    """.format(
            agent_descriptions=agent_descriptions.strip(),
            message=message.content,
            history=", ".join(msg.content for msg in history),
        )
        # logger.info(f"Planner prompt output: {planner_prompt}")
        return planner_prompt
