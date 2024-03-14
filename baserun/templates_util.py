from typing import Any, Dict, List, Optional, Union


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
