from ask_sdk_core.dispatch_components import AbstractExceptionHandler
from ask_sdk_core.dispatch_components import AbstractRequestHandler
from ask_sdk_core.skill_builder import SkillBuilder
from ask_sdk_core.handler_input import HandlerInput
from ask_sdk_model import Response
import ask_sdk_core.utils as ask_utils
import logging
import boto3

# Initialize the Amazon Bedrock client
bedrock_runtime = boto3.client('bedrock-runtime')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LaunchRequestHandler(AbstractRequestHandler):
    """Handler for Skill Launch."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_request_type("LaunchRequest")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Claude 3.5 mode activated"

        session_attr = handler_input.attributes_manager.session_attributes
        session_attr["chat_history"] = []

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class ClaudeQueryIntentHandler(AbstractRequestHandler):
    """Handler for Claude Query Intent."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return ask_utils.is_intent_name("GptQueryIntent")(handler_input)

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        query = handler_input.request_envelope.request.intent.slots["query"].value

        session_attr = handler_input.attributes_manager.session_attributes
        if "chat_history" not in session_attr:
            session_attr["chat_history"] = []

        response = generate_claude_response(
            session_attr["chat_history"], query)
        session_attr["chat_history"].append((query, response))

        return (
            handler_input.response_builder
            .speak(response)
            .ask("Any other questions?")
            .response
        )


class CatchAllExceptionHandler(AbstractExceptionHandler):
    """Generic error handling to capture any syntax or routing errors."""

    def can_handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> bool
        return True

    def handle(self, handler_input, exception):
        # type: (HandlerInput, Exception) -> Response
        logger.error(exception, exc_info=True)

        speak_output = "Sorry, I had trouble doing what you asked. Please try again."

        return (
            handler_input.response_builder
            .speak(speak_output)
            .ask(speak_output)
            .response
        )


class CancelOrStopIntentHandler(AbstractRequestHandler):
    """Single handler for Cancel and Stop Intent."""

    def can_handle(self, handler_input):
        # type: (HandlerInput) -> bool
        return (ask_utils.is_intent_name("AMAZON.CancelIntent")(handler_input) or
                ask_utils.is_intent_name("AMAZON.StopIntent")(handler_input))

    def handle(self, handler_input):
        # type: (HandlerInput) -> Response
        speak_output = "Leaving Claude 3.5 mode"

        return (
            handler_input.response_builder
            .speak(speak_output)
            .response
        )


def generate_claude_response(chat_history, new_question):
    """Generates a Claude 3.5 response to a new question using Amazon Bedrock"""
    messages = [{"role": "user", "content": [{"text": new_question}]}]

    # Add recent chat history (last 10 exchanges)
    for question, answer in chat_history[-10:]:
        # Insert earlier in the list so the new question remains at the end
        messages.insert(
            0, {"role": "assistant", "content": [{"text": answer}]})
        messages.insert(0, {"role": "user", "content": [{"text": question}]})

    try:
        # Using the converse API to communicate with Claude 3.5
        response = bedrock_runtime.converse(
            modelId='anthropic.claude-3-5-sonnet-20240620-v1:0',  # Claude 3.5 Sonnet model ID
            messages=messages,
            system=[
                {"text": "You are a helpful assistant. Answer in 50 words or less."}],
            inferenceConfig={
                'maxTokens': 300,
                'temperature': 0.5,
            }
        )

        # Extract the response text from the Claude model
        return response['output']['message']['content'][0]['text']

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"I encountered an error while processing your request. Please try again."


sb = SkillBuilder()

sb.add_request_handler(LaunchRequestHandler())
sb.add_request_handler(ClaudeQueryIntentHandler())
sb.add_request_handler(CancelOrStopIntentHandler())
sb.add_exception_handler(CatchAllExceptionHandler())

lambda_handler = sb.lambda_handler()
