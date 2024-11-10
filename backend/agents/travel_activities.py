import asyncio
import json
from typing import List

import aiohttp
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
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing_extensions import Annotated

from ..config import Config
from ..data_types import (
    Activities,
    AgentStructuredResponse,
    EndUserMessage,
    GroupChatMessage,
    HandoffMessage,
    TravelRequest,
)
from ..otlp_tracing import logger


# Retry logic for Bing search with exponential backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
async def _search_custom_bing(session, query_params: dict) -> dict:
    custom_config_id = Config.BING_CUSTOM_CONFIG_ID
    headers = {"Ocp-Apim-Subscription-Key": Config.BING_CUSTOM_SEARCH_KEY}
    async with session.get(
        url="https://api.bing.microsoft.com/v7.0/custom/search",
        params={
            "q": query_params["q"],
            "customConfig": custom_config_id,
            "count": query_params.get("count", 10),
            "freshness": query_params.get("freshness", "Month"),
        },
        headers=headers,
    ) as response:
        return await response.json()


# Fetch the content of a given URL and return the text
async def _fetch_content(session, url: str) -> str:
    async with session.get(url) as response:
        html_content = await response.text()
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator=" ", strip=True)


# Perform Bing search and extract content from the search results
async def get_info_from_bing_search(
    search_query: Annotated[str, "query to search on Bing for information"]
) -> str:
    logger.info(f"Performing Bing search for: {search_query}")
    async with aiohttp.ClientSession() as session:
        search_params = {"q": search_query, "count": 6}

        search_results = await _search_custom_bing(
            session=session, query_params=search_params
        )
        urls = [result["url"] for result in search_results["webPages"]["value"]]
        snippets = [result["snippet"] for result in search_results["webPages"]["value"]]

        # Limit the number of concurrent requests
        semaphore = asyncio.Semaphore(6)

        # Fetch content with semaphore to limit concurrency
        async def fetch_with_semaphore(url: str) -> str:
            async with semaphore:
                return await _fetch_content(session, url)

        tasks = [fetch_with_semaphore(url) for url in urls]
        contents = await asyncio.gather(*tasks)

        # Merge URLs, snippets, and contents into a single list of dictionaries
        merged_results = [
            {"url": url, "snippet": snippet, "content": content}
            for url, snippet, content in zip(urls, snippets, contents)
        ]
        logger.info(f"Search results: {merged_results}")
        return json.dumps(merged_results)


# Utility function to get travel activity tools
def get_travel_activity_tools() -> List[Tool]:
    return [
        FunctionTool(
            get_info_from_bing_search,
            description="This function performs a Bing search using the provided search query. It takes the search query as input and returns a list of dictionaries, each containing the URL, snippet, and content of a search result. This function is useful when the user wants to search for information on the web for travel activities.",
        )
    ]


# Activities Agent
@type_subscription("activities_booking")
class ActivitiesAgent(RoutedAgent):
    def __init__(
        self,
        model_client: AzureOpenAIChatCompletionClient,
        tools: List[Tool],
        tool_agent_type: str,
    ) -> None:
        super().__init__("ActivitiesAgent")
        self._system_messages: List[LLMMessage] = [
            SystemMessage(
                "You are a helpful AI assistant that can advise on activities."
            )
        ]
        self._model_client = model_client
        self._tools = tools
        self._tool_agent_id = AgentId(tool_agent_type, self.id.key)

    async def _process_request(
        self, message_content: str, ctx: MessageContext
    ) -> Activities:
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
        except Exception as e:
            logger.error(f"Tool agent caller loop failed: {str(e)}")
            return Activities(destination_city="", activities=[])

        # Ensure the final message content is a string
        assert isinstance(messages[-1].content, str)

        # Get structured data from the final message content
        try:
            response_content = await self._model_client.create(
                [UserMessage(content=messages[-1].content, source="ActivitiesAgent")],
                extra_create_args={"response_format": Activities},
            )
            return Activities.model_validate(json.loads(response_content.content))
        except Exception as e:
            logger.error(f"Failed to parse activities response: {str(e)}")
            return Activities(destination_city="", activities=[])

    @message_handler
    async def handle_message(
        self, message: EndUserMessage, ctx: MessageContext
    ) -> None:
        if "travel plan" in message.content.lower():
            # Cannot handle complex travel plans, hand off back to router
            await self.publish_message(
                HandoffMessage(content=message.content, source=self.id.type),
                DefaultTopicId(type="router", source=ctx.topic_id.source),
            )
            return

        activities_structured = await self._process_request(message.content, ctx)

        # Publish the response to the group chat manager
        await self.publish_message(
            AgentStructuredResponse(
                agent_type=self.id.type,
                data=activities_structured,
                message=f"Activities processed successfully for query - {message.content}",
            ),
            DefaultTopicId(type="user_proxy", source=ctx.topic_id.source),
        )

    @message_handler
    async def handle_travel_request(
        self, message: TravelRequest, ctx: MessageContext
    ) -> GroupChatMessage:
        activities_structured = await self._process_request(message.content, ctx)

        # send the response to the group chat manager
        return GroupChatMessage(
            source=self.id.type,
            content=activities_structured.model_dump_json(),
        )
