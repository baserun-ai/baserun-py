import logging
import re
from typing import Any, Dict, List, Optional, Union

from baserun.v1.baserun_pb2 import InputVariable, Message, Span

logger = logging.getLogger(__name__)


class FormattedTemplateData:
    def __init__(
        self,
        variables: Dict[str, Any],
        template_id: str,
        formatted_messages: Optional[List[Dict[str, Union[str, Dict[str, Any]]]]] = None,
    ) -> None:
        self.variables = variables
        self.template_id = template_id
        self.formatted_messages = formatted_messages


class FormattedContentString(str):
    template_data: FormattedTemplateData

    def __new__(cls, val: str, template_data: FormattedTemplateData):
        inst = super().__new__(cls, val)
        inst.template_data = template_data
        return inst


def find_template_match(messages: List[Message]) -> Union[str, None]:
    from baserun import Baserun

    if not messages:
        return None

    message_contents = [message.content for message in messages]

    if Baserun.formatted_templates is None:
        logger.warning("Baserun attempted to submit span, but baserun.init() was not called")
        return None

    for template_name, formatted_templates in Baserun.formatted_templates.items():
        for formatted_template in formatted_templates:
            # FIXME? What if there are multiple matches? Maybe check for the highest # of messages
            if all(message in message_contents for message in formatted_template):
                return template_name

    return None


def match_messages_to_template(span: Span):
    from baserun import Baserun, get_template

    prompt_messages = span.prompt_messages
    matched_template = find_template_match(list(prompt_messages))
    if matched_template and (template := get_template(matched_template)):
        annotation = Baserun.annotate(span.completion_id)
        span.template_id = template.id

        if template.active_version:
            template_variables = {}
            for message in prompt_messages:
                for template_message in template.active_version.template_messages:
                    template_variables.update(reverse_engineer_variables(template_message.message, message.content))

            for key, value in template_variables.items():
                input_variable = InputVariable(key=key, value=value)
                annotation.input_variables.append(input_variable)

        annotation.submit()


def reverse_engineer_variables(template: str, formatted_string: str) -> Dict[str, str]:
    # Escape any characters in the template that might be interpreted as regex special characters, except for the
    # variable braces {}.
    template_escaped = re.sub(r"([\[\].*+?^=!:${}()|\[\]\\/])", r"\\\1", template)

    # Convert the template into a regex pattern, turning {variable} into named regex groups.
    pattern = re.sub(r"\\{([a-zA-Z0-9_-]+)\\}", r"(?P<\1>[^,]+)", template_escaped)

    # Use the regex pattern to search the formatted string.
    match = re.match(pattern, formatted_string)
    if match:
        # Extract the variable values from the match.
        return match.groupdict()
    else:
        return {}
