import hashlib
import inspect
import logging
import os
import sys
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from baserun import Baserun
from baserun.grpc import (
    get_or_create_async_submission_service,
    get_or_create_submission_service,
)
from baserun.helpers import memoize_for_time
from baserun.templates_util import FormattedContentString, FormattedTemplateData
from baserun.v1.baserun_pb2 import (
    GetTemplatesRequest,
    GetTemplatesResponse,
    SubmitTemplateVersionRequest,
    SubmitTemplateVersionResponse,
    Template,
    TemplateMessage,
    TemplateVersion,
)

if TYPE_CHECKING:
    # Just for type annotations for Langchain. Since langchain is an optional dependency we have this garbage
    try:
        from langchain.tools import Tool
    except ImportError:
        pass

logger = logging.getLogger(__name__)


@memoize_for_time(os.environ.get("BASERUN_CACHE_INTERVAL", 600))
def get_templates(environment: Union[str, None] = None) -> Dict[str, Template]:
    try:
        request = GetTemplatesRequest(environment=environment or Baserun.environment)
        response: GetTemplatesResponse = get_or_create_submission_service().GetTemplates(request)
        for template in response.templates:
            Baserun.templates[template.name] = template

    except BaseException:
        logger.error(f"Could not fetch templates from Baserun. Using {len(Baserun.templates.keys())} cached templates")
        exc_type, exc_value, exc_traceback = sys.exc_info()
        logger.info(traceback.format_exception(exc_type, exc_value, exc_traceback))

    return Baserun.templates


def get_template(name: str) -> Union[Template, None]:
    templates = get_templates()
    template = templates.get(name)
    if not template:
        logger.info(f"Attempted to get template {name} but no template with that name exists")
        return None

    return template


def get_template_type_enum(template_type: Optional[str] = None):
    template_type = template_type or "unknown"
    if template_type == Template.TEMPLATE_TYPE_JINJA2 or template_type.lower().startswith("jinja"):
        template_type_enum = Template.TEMPLATE_TYPE_JINJA2
    else:
        template_type_enum = Template.TEMPLATE_TYPE_FORMATTED_STRING

    return template_type_enum


def apply_template(
    template_name: str,
    parameters: Dict[str, Any],
    template_messages: List[Dict[str, Union[str, Dict[str, Any]]]],
    template_type_enum: Template.TemplateType,
) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
    # Ensure the latest templates used in production are synced to the UI
    template = register_template(
        template_messages=template_messages,
        template_name=template_name,
        template_type=template_type_enum,
    )

    formatted_messages = []
    template_data = FormattedTemplateData(parameters, template_id=template.id)

    for message in template_messages:
        template_string = message.get("content")
        if isinstance(template_string, str):
            if template_type_enum == Template.TEMPLATE_TYPE_JINJA2:
                try:
                    # noinspection PyUnresolvedReferences
                    from jinja2 import Template as JinjaTemplate

                    jinja_template = JinjaTemplate(template_string)
                    formatted_content = jinja_template.render(parameters)
                except ImportError:
                    logger.warning("Cannot render Jinja2 template as jinja2 package is not installed")
                    # TODO: Is this OK? should we raise? or return blank string?
                    formatted_content = template_string
            else:
                try:
                    formatted_content = template_string.format(**parameters)
                except KeyError:
                    formatted_content = template_string

            formatted_messages.append(
                {
                    **message,
                    "content": FormattedContentString(formatted_content, template_data),
                }
            )
        else:
            formatted_messages.append(message)

    template_data.formatted_messages = formatted_messages
    formatted_template_list = Baserun.formatted_templates.get(template_name)
    set_value: Tuple[str, ...] = tuple([str(message.get("content")) for message in formatted_messages])

    if formatted_template_list:
        formatted_template_list.add(set_value)
    else:
        formatted_template_list = {set_value}

    Baserun.formatted_templates[template_name] = formatted_template_list

    return formatted_messages


def create_langchain_template(
    template_string: str,
    parameters: Optional[Dict[str, Any]] = None,
    template_name: Optional[str] = None,
    template_tag: Optional[str] = None,
    template_type: Optional[str] = None,
    tools: Optional[List[Union["Tool", Any]]] = None,
):
    from langchain.prompts import ChatPromptTemplate
    from langchain_core.messages import BaseMessage

    parameters = parameters or {}
    input_variables = list(parameters.keys())
    template_type = template_type or "Formatted String"
    tools = tools or []

    if not template_name:
        caller = inspect.stack()[1].function
        template_name = f"{caller}_template"

    langchain_template = ChatPromptTemplate(
        messages=[BaseMessage(role="SYSTEM", content=template_string)],
        input_variables=input_variables,
    )

    template_type_enum = get_template_type_enum(template_type)
    register_template(
        template_messages=[{"role": "SYSTEM", "content": template_string}],
        template_name=template_name,
        template_type=template_type_enum,
        template_tag=template_tag,
    )

    return langchain_template


def format_prompt(
    template_name: str,
    parameters: Dict[str, Any],
    template_messages: Optional[List[Dict[str, Union[str, Dict[str, Any]]]]] = None,
    template_type: Optional[str] = None,
    submit_variables: bool = True,  # unused, left for compatibility
) -> List[Dict[str, Union[str, Dict[str, Any]]]]:
    template_type_enum = get_template_type_enum(template_type)
    template = get_template(template_name)

    if template:
        # if submit_variables:
        # for key, value in parameters.items():
        #     baserun.submit_input_variable(key=key, value=value, template=template)

        if not template_messages:
            template_messages = [
                {"role": message.role, "content": message.message}
                for message in template.active_version.template_messages
            ]

    if not template_messages:
        template_messages = []

    return apply_template(
        template_name=template_name,
        parameters=parameters,
        template_messages=template_messages,
        template_type_enum=template_type_enum,
    )


def construct_template_version(
    template_messages: List[Dict[str, Union[str, Dict[str, Any]]]],
    template_name: Optional[str] = None,
    template_tag: Optional[str] = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
) -> TemplateVersion:
    # Automatically generate a name based on the template's contents
    if not template_name:
        template_name = hashlib.sha256(
            "".join([str(message.get("content")) for message in template_messages]).encode()
        ).hexdigest()[:5]

    template = Template(name=template_name, template_type=template_type)
    constructed_template_messages = [
        TemplateMessage(
            role=str(message.get("role", "assistant")),
            message=str(message.get("message", message.get("content"))),
            order_index=i,
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
    template_messages: List[Dict[str, Union[str, Dict[str, Any]]]],
    template_name: str,
    template_tag: Optional[str] = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
) -> Template:
    # TODO: if template already exists you'd probably want to create new version of it instead of doing nothing
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

    template = response.template_version.template
    if template.name not in Baserun.templates:
        Baserun.templates[template.name] = template

    return template


async def aregister_template(
    template_messages: List[Dict[str, Union[str, Dict[str, Any]]]],
    template_name: str,
    template_tag: Optional[str] = None,
    template_type=Template.TEMPLATE_TYPE_FORMATTED_STRING,
) -> Template:
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
