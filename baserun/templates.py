import hashlib
import inspect
import logging
import os
import traceback
from typing import Any, TYPE_CHECKING, Union, Set

from baserun import Baserun
from baserun.grpc import (
    get_or_create_submission_service,
    get_or_create_async_submission_service,
)
from baserun.helpers import memoize_for_time
from baserun.v1.baserun_pb2 import (
    Template,
    TemplateVersion,
    SubmitTemplateVersionRequest,
    SubmitTemplateVersionResponse,
    GetTemplatesRequest,
    GetTemplatesResponse,
    TemplateMessage,
)

if TYPE_CHECKING:
    # Just for type annotations for Langchain. Since langchain is an optional dependency we have this garbage
    try:
        from langchain.tools import Tool
    except ImportError:
        Tool = Any

logger = logging.getLogger(__name__)


@memoize_for_time(os.environ.get("BASERUN_CACHE_INTERVAL", 600))
def get_templates(environment: str = None) -> dict[str, Template]:
    if not Baserun.templates:
        Baserun.templates = {}

    try:
        request = GetTemplatesRequest(environment=environment or Baserun.environment)
        response: GetTemplatesResponse = get_or_create_submission_service().GetTemplates(request)
        for template in response.templates:
            Baserun.templates[template.name] = template

    except BaseException as e:
        logger.error(f"Could not fetch templates from Baserun. Using {len(Baserun.templates.keys())} cached templates")
        logger.info(traceback.format_exception(e))

    return Baserun.templates


def get_template(name: str) -> Template:
    templates = get_templates()
    template = templates.get(name)
    if not template:
        logger.info(f"Attempted to get template {name} but no template with that name exists")
        return None

    return template


def get_template_type_enum(template_type: str = None):
    template_type = template_type or "unknown"
    if template_type == Template.TEMPLATE_TYPE_JINJA2 or template_type.lower().startswith("jinja"):
        template_type_enum = Template.TEMPLATE_TYPE_JINJA2
    else:
        template_type_enum = Template.TEMPLATE_TYPE_FORMATTED_STRING

    return template_type_enum


def apply_template(
    template_name: str,
    template_messages: list[dict[str, Union[str, dict[str, Any]]]],
    parameters: dict[str, Any],
    template_type_enum,
) -> str:
    formatted_messages = []
    for message in template_messages:
        template_string = message.get("content")
        if template_string:
            if template_type_enum == Template.TEMPLATE_TYPE_JINJA2:
                try:
                    # noinspection PyUnresolvedReferences
                    from jinja2 import Template as JinjaTemplate

                    template = JinjaTemplate(template_string)
                    return template.render(parameters)
                except ImportError:
                    logger.warning("Cannot render Jinja2 template as jinja2 package is not installed")
                    # TODO: Is this OK? should we raise? or return blank string?
                    return template_string

            formatted_content = template_string.format(**parameters)
            formatted_messages.append({**message, "content": formatted_content})
        else:
            formatted_messages.append(message)

    formatted_template_list: Set[tuple] = Baserun.formatted_templates.get(
        template_name,
    )
    set_value = tuple([message.get("content") for message in formatted_messages])

    if not formatted_template_list:
        formatted_template_list = {set_value}
    else:
        formatted_template_list.add(set_value)

    Baserun.formatted_templates[template_name] = formatted_template_list
    return formatted_messages


def create_langchain_template(
    template_string: str,
    parameters: dict[str, Any] = None,
    template_name: str = None,
    template_tag: str = None,
    template_type: str = None,
    tools: list[Union["Tool", Any]] = None,
):
    from langchain.prompts import PromptTemplate

    parameters = parameters or {}
    input_variables = list(parameters.keys())
    template_type = template_type or "Formatted String"
    tools = tools or []

    if not template_name:
        caller = inspect.stack()[1].function
        template_name = f"{caller}_template"

    if template_type == Template.TEMPLATE_TYPE_JINJA2 or template_type.lower().startswith("jinja"):
        langchain_template = PromptTemplate(
            template=template_string,
            input_variables=input_variables,
            template_format="jinja2",
            tools=tools,
        )

    else:
        langchain_template = PromptTemplate(template=template_string, input_variables=input_variables, tools=tools)

    template_type_enum = get_template_type_enum(template_type)
    template_version = register_template(
        template_string=template_string,
        template_name=template_name,
        template_type=template_type_enum,
        template_tag=template_tag,
    )

    return langchain_template


def format_prompt(
    template_name: str,
    parameters: dict[str, Any],
    template_messages: list[dict[str, Union[str, dict[str, Any]]]] = None,
    template_type: str = None,
):
    template_type_enum = get_template_type_enum(template_type)
    if not template_messages:
        template = get_template(template_name)
        template_messages = [
            {"role": message.role, "content": message.message} for message in template.active_version.template_messages
        ]

    return apply_template(template_name, template_messages, parameters, template_type_enum)


def construct_template_version(
    template_messages: list[dict[str, Union[str, dict[str, Any]]]],
    template_name: str = None,
    template_tag: str = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
) -> TemplateVersion:
    # Automatically generate a name based on the template's contents
    if not template_name:
        template_name = hashlib.sha256(
            "".join([message.get("content") for message in template_messages]).encode()
        ).hexdigest()[:5]

    template = Template(name=template_name, template_type=template_type)
    constructed_template_messages = [
        TemplateMessage(
            role=message.get("role", "assistant"), message=message.get("message", message.get("content")), order_index=i
        )
        for i, message in enumerate(template_messages)
    ]
    version = TemplateVersion(
        template=template,
        template_messages=constructed_template_messages,
        tag=template_tag,
    )

    return version


def register_template(
    template_messages: list[dict[str, Union[str, dict[str, Any]]]],
    template_name: str = None,
    template_tag: str = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
) -> Template:
    from baserun import Baserun

    if not Baserun.templates:
        Baserun.templates = {}

    if template := Baserun.templates.get(template_name):
        return template

    version = construct_template_version(
        template_messages=template_messages,
        template_name=template_name,
        template_tag=template_tag,
        template_type=template_type,
    )

    request = SubmitTemplateVersionRequest(template_version=version, environment=Baserun.environment)
    response: SubmitTemplateVersionResponse = get_or_create_submission_service().SubmitTemplateVersion(request)

    template = response.template
    if template.name not in Baserun.templates:
        Baserun.templates[template.name] = response.template

    return response.template


async def aregister_template(
    template_messages: list[dict[str, Union[str, dict[str, Any]]]],
    template_name: str = None,
    template_tag: str = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
) -> Template:
    from baserun import Baserun

    if not Baserun.templates:
        Baserun.templates = {}

    if template := Baserun.templates.get(template_name):
        return template

    version = construct_template_version(
        template_messages=template_messages,
        template_name=template_name,
        template_tag=template_tag,
        template_type=template_type,
    )

    request = SubmitTemplateVersionRequest(template_version=version, environment=Baserun.environment)
    response: SubmitTemplateVersionResponse = await get_or_create_async_submission_service().SubmitTemplateVersion(
        request
    )

    template = response.template
    if template.name not in Baserun.templates:
        Baserun.templates[template.name] = template

    return template
