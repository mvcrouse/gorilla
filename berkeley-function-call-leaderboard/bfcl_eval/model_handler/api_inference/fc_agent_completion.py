import os
import time

import requests
from openai import RateLimitError

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    convert_to_tool,
    decoded_output_to_execution_list,
    retry_with_backoff,
)


###
#
###
class FCAgentCompletionsHandler(BaseHandler):
    def __init__(
        self,
        *args,
        agent_params: dict | None = None,
        agent_url: str = "http://localhost:8001/chat",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS

        self._agent_url = os.getenv("FC_AGENT_AGENT_URL") or agent_url
        self._agent_params = agent_params or dict()
        self._agent_params["model_params"] = {
            "temperature": self.temperature,
            **self._agent_params.get("model_params", dict()),
        }

    def decode_ast(self, result, language, has_tool_call_tag):
        decoded_output = []
        for invoked_function in result:
            name = invoked_function["name"]
            params = invoked_function["arguments"]
            decoded_output.append({name: params})
        return decoded_output

    def decode_execute(self, result, has_tool_call_tag):
        decoded = self.decode_ast(
            result, language=None, has_tool_call_tag=has_tool_call_tag
        )
        return decoded_output_to_execution_list(decoded)

    @retry_with_backoff(error_type=RateLimitError)
    def generate_with_backoff(self, **kwargs):
        start_time = time.time()
        response = requests.post(
            url=self._agent_url, json={**self._agent_params, **kwargs}
        ).json()
        end_time = time.time()
        return response, end_time - start_time

    #### FC methods ####

    def _query_prompting(self, inference_data: dict):
        return self._query_FC(inference_data)

    def _query_FC(self, inference_data: dict):
        messages: list[dict] = inference_data["messages"]

        tools = inference_data["tools"]
        inference_data["inference_input_log"] = {
            "messages": repr(messages),
            "tools": tools,
        }
        kwargs = {"messages": messages, "tools": tools}

        return self.generate_with_backoff(**kwargs)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        inference_data["messages"] = []
        return inference_data

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, self.model_style)
        tools = [t.get("function") for t in tools]

        inference_data["tools"] = tools

        return inference_data

    def _parse_query_response_FC(self, api_response: list[dict]) -> dict:
        response = api_response[-1]
        model_responses = response.get("tool_calls") or response.get("content")
        tool_call_ids = (
            [func_call.get("id") for func_call in model_responses]
            if isinstance(model_responses, list)
            else model_responses
        )
        return {
            "model_responses": model_responses,
            "model_responses_message_for_chat_history": api_response,
            "tool_call_ids": tool_call_ids,
            "input_token": 0,
            "output_token": 0,
        }

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["messages"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_FC(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["messages"].extend(user_message)
        return inference_data

    def _add_assistant_message_FC(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["messages"].extend(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_FC(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        # Add the execution results to the current round result, one at a time
        for execution_result in execution_results:
            tool_message = {"role": "tool", "content": execution_result}
            inference_data["messages"].append(tool_message)

        return inference_data
