from autogen_core.application import SingleThreadedAgentRuntime
from autogen_core.base import AgentId
from autogen_core.components import DefaultSubscription
from autogen_core.components.tool_agent import ToolAgent
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import ChatSummaryMemoryBuffer
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.tools.wikipedia import WikipediaToolSpec
from azure.messaging.webpubsubservice import WebPubSubServiceClient

from .agents.ext_agents import LlamaIndexAgent
from .agents.travel_activities import ActivitiesAgent, get_travel_activity_tools
from .agents.travel_car import CarRentalAgent
from .agents.travel_destination import DestinationAgent
from .agents.travel_flight import FlightAgent
from .agents.travel_group_chat import GroupChatManager
from .agents.travel_hotel import HotelAgent, get_hotel_booking_tool
from .agents.travel_router import SemanticRouterAgent
from .config import Config
from .otlp_tracing import configure_oltp_tracing, logger
from .registry import AgentRegistry
from .session_state import SessionStateManager

# Create Wikipedia tool specification
wiki_spec = WikipediaToolSpec()
wikipedia_tool = wiki_spec.to_tool_list()[1]

# Configure tracing
tracer = configure_oltp_tracing()

# Create AzureOpenAI model instance
llm = AzureOpenAI(
    deployment_name=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
    temperature=0.01,
    max_tokens=2000,
    api_key=Config.AZURE_OPENAI_API_KEY,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    api_version=Config.AZURE_OPENAI_API_VERSION,
)

model_capabilities = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
}
aoai_model_client = Config.GetAzureOpenAIChatCompletionClient(model_capabilities)

session_state_manager = SessionStateManager()


async def initialize_agent_runtime() -> SingleThreadedAgentRuntime:
    """
    Initializes the agent runtime with the required agents and tools.

    Returns:
        SingleThreadedAgentRuntime: The initialized runtime for managing agents.
    """
    global session_state_manager, aoai_model_client
    agent_runtime = SingleThreadedAgentRuntime(tracer_provider=tracer)

    agent_registry = AgentRegistry()
    travel_activity_tools = get_travel_activity_tools()
    hotel_booking_tool = get_hotel_booking_tool()

    # Register Tool Agents
    await ToolAgent.register(
        agent_runtime,
        "activity_tool_executor_agent",
        lambda: ToolAgent("Travel tool executor agent", travel_activity_tools),
    )
    await ToolAgent.register(
        agent_runtime,
        "hotel_booking_tool_exec_agent",
        lambda: ToolAgent("Hotel tool executor agent", hotel_booking_tool),
    )

    # Add subscriptions
    await agent_runtime.add_subscription(
        DefaultSubscription(topic_type="user_proxy", agent_type="user_proxy")
    )

    # Register Semantic Router Agent
    await SemanticRouterAgent.register(
        agent_runtime,
        "router",
        lambda: SemanticRouterAgent(
            name="SemanticRouterAgent",
            model_client=aoai_model_client,
            agent_registry=agent_registry,
            session_manager=session_state_manager,
        ),
    )

    # Register other agents
    await FlightAgent.register(agent_runtime, "flight_booking", lambda: FlightAgent())
    await HotelAgent.register(
        agent_runtime,
        "hotel_booking",
        lambda: HotelAgent(
            aoai_model_client, hotel_booking_tool, "hotel_booking_tool_exec_agent"
        ),
    )
    await CarRentalAgent.register(agent_runtime, "car_rental", lambda: CarRentalAgent())
    await ActivitiesAgent.register(
        agent_runtime,
        "activities_booking",
        lambda: ActivitiesAgent(
            aoai_model_client, travel_activity_tools, "activity_tool_executor_agent"
        ),
    )
    await DestinationAgent.register(
        agent_runtime, "destination_info", lambda: DestinationAgent(aoai_model_client)
    )
    await LlamaIndexAgent.register(
        agent_runtime,
        "default_agent",
        lambda: LlamaIndexAgent(
            llama_index_agent=ReActAgent.from_tools(
                tools=[wikipedia_tool],
                llm=llm,
                max_iterations=5,
                memory=ChatSummaryMemoryBuffer(llm=llm, token_limit=1000),
            ),
        ),
    )
    await GroupChatManager.register(
        agent_runtime, "group_chat_manager", lambda: GroupChatManager()
    )

    # Start the runtime
    agent_runtime.start()

    logger.info("Agent runtime initialized successfully.")

    return agent_runtime


def get_web_pub_client() -> WebPubSubServiceClient:
    service = WebPubSubServiceClient.from_connection_string(
        Config.WEB_PUB_SUB_CONNECTION_STRING, hub=Config.WEB_PUB_SUB_HUB_NAME
    )
    return service
