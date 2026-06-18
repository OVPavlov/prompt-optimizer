import os
from pathlib import Path
from .LLMClient import LLModel
from .PromptCommon import ParseError, extract_tag, safe_format, read_prompting_template, norm_f_name


class MetaPrompt:
	def __init__(self, dir, llmodel:LLModel, task_description:str):
		self.dir = Path(dir)
		self.llmodel:LLModel = llmodel
		self.task_description = task_description

		self.write_file("task_description", task_description)
		self.output_format_spec = None
		self.schema_design = None
		self.analysis_guidance = None
		self.generation_guidance = None
		self.analysis_system_message = None
		self.no_fluff = ''


	def write_file(self, file_name:str, content:str):
		os.makedirs(self.dir, exist_ok=True)
		(self.dir / norm_f_name(file_name)).write_text(content)

	def read_file(self, file_name:str):
		path = self.dir / norm_f_name(file_name)
		if not path.exists():
			return None
		return path.read_text()

	def generate_rating_schema(self):
		user_prompt = f"## Task Description\n{self.task_description}"

		rating_schema_instructions = read_prompting_template("rating-schema-instructions")
		po_sys_description = read_prompting_template("prompt-optimization-system-description.xml")
		self.no_fluff = read_prompting_template('no_fluff.txt')
		rating_schema_source = self.read_file("rating_schema_source")
		sys_prompt = f"""{self.no_fluff}\n## System Description
{po_sys_description}
## Task Description
<task_description>
{self.task_description}
</task_description>
---

## Instructions
{rating_schema_instructions}"""

		analysis_template = read_prompting_template("analysis_template")

		if rating_schema_source is None:
			rating_schema_source = self.llmodel.request(sys_prompt, user_prompt)
			self.write_file("rating_schema_source", rating_schema_source)

		try:
			self.output_format_spec = extract_tag(rating_schema_source, 'output_format_spec')
			self.schema_design = extract_tag(rating_schema_source, 'schema_design')
			self.analysis_guidance = extract_tag(rating_schema_source, 'analysis_guidance')
			self.generation_guidance = extract_tag(rating_schema_source, 'generation_guidance')
		except:
			raise ParseError(system=sys_prompt, user=user_prompt, output=rating_schema_source, model=self.llmodel.model_id,
							 task="Generate rating schema", failure="Failed to parse tags")
		try:
			self.analysis_system_message = safe_format(analysis_template, vars(self))
		except:
			raise ParseError(system=sys_prompt, user=user_prompt, output=rating_schema_source, model=self.llmodel.model_id,
							 task="Generate rating schema", failure="Failed to format analysis_template")



