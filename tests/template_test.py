from uuid import uuid4

import pytest

from baserun import format_prompt, Baserun
from baserun.templates import get_template
from baserun.v1.baserun_pb2 import Template, TemplateVersion, GetTemplatesResponse


@pytest.fixture(autouse=True)
def clear_registered_templates():
    Baserun.templates = {}


def test_format_prompt_string(mock_services):
    template = "Hello {name}"
    name = "Jimothy"
    template_name = "my_template"
    formatted_template = format_prompt(
        template_string=template,
        parameters={"name": name},
        template_name=template_name,
    )

    assert formatted_template == f"Hello {name}"

    mock_submit_template_version = mock_services["submission_service"].SubmitTemplateVersion
    assert mock_submit_template_version.call_count == 1
    args, kwargs = mock_submit_template_version.call_args_list[0]

    request = args[0]
    assert request.template_version.template.name == template_name
    assert request.template_version.template.template_type == 1
    assert request.template_version.tag == "83358"
    assert request.template_version.parameter_definition == '{"name": "string"}'


def test_format_prompt_jinja2(mock_services):
    template = "Hello {{name}}"
    name = "Jimothy"
    template_name = "my_template"
    formatted_template = format_prompt(
        template_string=template,
        template_name=template_name,
        template_type="Jinja2",
        parameters={"name": name},
    )

    assert formatted_template == f"Hello {name}"

    mock_submit_template_version = mock_services["submission_service"].SubmitTemplateVersion
    assert mock_submit_template_version.call_count == 1
    args, kwargs = mock_submit_template_version.call_args_list[0]

    request = args[0]
    assert request.template_version.template.name == template_name
    assert request.template_version.template.template_type == 2
    assert request.template_version.tag == "652b7"
    assert request.template_version.parameter_definition == '{"name": "string"}'


def test_get_template(mock_services):
    template_name = "my_template"
    template_string = "Hello {{name}}"
    tag = "latest"

    template = Template(id=str(uuid4()), name=template_name)
    template_version = TemplateVersion(id=str(uuid4()), tag=tag, template_string=template_string)
    template.active_version.CopyFrom(template_version)
    template_version.template.CopyFrom(template)
    template.template_versions.extend([template_version])

    mock_get_templates = mock_services["submission_service"].GetTemplates
    mock_get_templates.return_value = GetTemplatesResponse(templates=[template])

    found_template = get_template(name=template_name)
    found_template_version = found_template.active_version

    assert found_template.name == template_name
    assert found_template.id == template.id
    assert found_template_version.id == template_version.id
    assert found_template_version.tag == tag
