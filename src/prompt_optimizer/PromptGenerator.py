from .LLMClient import LLModel
from .DataTypes import Prompt, ModelResult, IterationInfo, ResultDataset, ExperimentValues
from .PromptCommon import ParseError, get_ratings, extract_tag, extract_all, safe_format
from .MetaPrompt import MetaPrompt



def get_system_prompt_template(first_prompt:bool, one_model:bool, meta_prompt:MetaPrompt) -> str:
	template = """
{no_fluff}

You are optimizing a prompt for an executor LLM performing the following task:

<task_description>
{task_description}
</task_description>

How to address each rating dimension:

{generation_guidance}

---
"""
	if first_prompt:
		template += """

Produce executor prompt to achieve the task.

**Prompt hygiene:** Prefer clarity over added constraints.
	- Once a problem is resolved, remove the constraint that fixed it. Accumulated rules make prompts brittle.

**Strictly follow formatting rules.**

Output exactly this tag:
<system_prompt>
Executor system prompt.
</system_prompt>
"""
		return safe_format(template, vars(meta_prompt))

	if one_model:
		template += """
You will receive the full iteration history: each iteration's system prompt, and prompt performance analysis summary.
You will also receive the last iteration's system prompt, analysis and ratings"""
	else:
		template += """
You will receive the full iteration history: each iteration's system prompt, per-model instructions, and prompt performance analysis summary.
You will also receive the last iteration's system prompt, per-model instructions, and per-model analysis and ratings.
Models are anonymized as letters (A, B, ...).
"""

	template += """
1. Summerize the analysis for the last iteration and assess the performance of the last prompt into the last_iteration_summary tag.
2. Produce the improved executor prompt to achieve the task.
3. Write very short and very concise list of the changes made to the prompt into prompt_changes tag.
"""
	if not one_model:
		template += """
**Template-literal constraint:**
Your <system_prompt> is executed as: system_prompt.format(per_model_instructions=...)
Therefore:
- {per_model_instructions} must appear as a literal string in <system_prompt> wherever per-model instructions should be injected. Do not resolve it.

**Model instructions:** Only add <per_model_instructions> when a specific model needs adjustment not addressable in the shared prompt.
	- Use the same letter IDs as appear in the history.

"""
	template += """

**Prompt hygiene:** Prefer clarity over added constraints.
	- Once a problem is resolved, remove the constraint that fixed it. Accumulated rules make prompts brittle.

**Scope:** You see only analyzer prose and ratings — not raw data or raw executor outputs.
	- If the analysis is insufficient to diagnose a problem, do not guess at a fix.


**Strictly follow formatting rules.**

Output exactly these tags:

<last_iteration_summary>
Summary of the last iteration's analysis, ratings, and performance of the last prompt.
</last_iteration_summary>

<system_prompt>
Executor system prompt. Include literal {per_model_instructions} if per-model instructions are used.
</system_prompt>

<prompt_changes>
Prose, very short and very concise list of the changes made to the prompt.
</prompt_changes>
"""
	if not one_model:
		template += """
<per_model_instructions>
<model id="A">...</model>
</per_model_instructions>
"""

	return safe_format(template, vars(meta_prompt))








def get_summary_context(dataset:ResultDataset, models_anon:dict, iteration:int):
	it_info:IterationInfo = dataset.get_i(iteration)
	model_num = len(it_info.models)
	prompt:Prompt = it_info.prompt
	system_prompt = prompt.system_message_for(it_info.models[0]) if model_num == 0 else prompt.system
	iteration_summary_format = """
<iteration num={info.iteration}>
<system_prompt>{system_prompt}</system_prompt>"""
	if it_info.prompt.user_message is not None:
		iteration_summary_format += "\n<user_message>{info.prompt.user_message}</user_message>"

	if model_num > 1 and prompt.instructions is not None and len(prompt.instructions) > 0:
		mi = '\n'.join([f'<model id="{models_anon[k]}">{v}</model>' for k, v in prompt.instructions.items()])
		iteration_summary_format += f"""
<per_model_instructions>
{mi}
</per_model_instructions>
"""
	iteration_summary_format += """
<prompt_changes_summary>
{info.prompt_changes}
</prompt_changes_summary>

<analysis_summary>
{info.analysis_summary}
</analysis_summary>
<ratings>
{ratings}
</ratings>
</iteration>
"""
	ratings = get_ratings(dataset, iteration)
	if iteration == 0:
		ratings = "\n".join(f" - {k}: {v:.0%}" for k, v in ratings.items())
	else:
		rb = get_ratings(dataset, iteration-1)
		ratings = "\n".join(f" - {k}: {v:.0%} ({(v - rb[k]):+.0%})" for k, v in ratings.items())

	return iteration_summary_format.format(info=dataset.get_i(iteration), ratings=ratings, system_prompt=system_prompt)


def model_format(mr:ModelResult, models_anon:dict, multiple:bool):
	if multiple:
		m_f = """
<model id="{letter}">
<per_model_instructions>{res.prompt.instructions}</per_model_instructions>
<analysis>{res.analysis}</analysis>
<rating>{res.rating}</rating>
</model>
"""
	else:
		m_f = """
<model id="{letter}">
<analysis>{res.analysis}</analysis>
<rating>{res.rating}</rating>
</model>
"""
	return m_f.format(letter=models_anon[mr.model], res=mr)

def iteration_format(info: IterationInfo, analysis:str):
	i_form = """
<iteration num={info.iteration}>
<system_prompt>{info.prompt.system}</system_prompt>"""
	if info.prompt.user_message is not None:
		i_form += "\n<user_message>{info.prompt.user_message}</user_message>"
	i_form += """
<models_analysis>{analysis}</models_analysis>
</iteration>
"""
	return i_form.format(info=info, analysis=analysis)

def get_full_context(dataset:ResultDataset, models_anon:dict, iteration:int):
	per_model_context = []
	iteration_info: IterationInfo = dataset.get_i(iteration)
	multiple:bool = len(iteration_info.models) > 1
	for model in iteration_info.models:
		res = dataset.get_mr(model, iteration)
		per_model_context.append(model_format(res, models_anon, multiple=multiple))
	model_analysis = '\n'.join(per_model_context)
	return iteration_format(info=iteration_info, analysis=model_analysis)




def _parse_response(response:str, from_anon:dict) -> Prompt:
	system_prompt = extract_tag(response, 'system_prompt')
	user_prompt = extract_tag(response, 'user_prompt')
	instructions_text = extract_tag(response, 'per_model_instructions')
	instructions_dict = {}
	if instructions_text is not None:
		for t in extract_all(instructions_text, 'model'):
			instructions_dict[from_anon[t.attrs['id']]] = (t.body or '').strip()

	return Prompt(system_prompt, instructions_dict, user_prompt)


class PromptGenerator:
	def __init__(self, meta_prompt:MetaPrompt, user_message:str, llmodel:LLModel):
		self.first_system_prompt = get_system_prompt_template(first_prompt=True, one_model=False, meta_prompt=meta_prompt)
		self.system_prompt = get_system_prompt_template(first_prompt=False, one_model=False, meta_prompt=meta_prompt)
		self.one_model_system_prompt = get_system_prompt_template(first_prompt=False, one_model=True, meta_prompt=meta_prompt)
		self.user_message = user_message
		self.llmodel:LLModel = llmodel

	def generate_prompt(self, dataset:ResultDataset, values:ExperimentValues) -> Prompt:
		context = PromptGenerator.get_all_models_context(dataset, values.to_anon)
		user_message = str(context) if self.user_message is None else self.user_message.format(context=context)

		if len(values.current_models) == 1:
			system_prompt = self.one_model_system_prompt
		else:
			system_prompt = self.system_prompt

		response = self.llmodel.request(system_prompt, user_message)
		with ParseError.guard(system_prompt, user_message, response, self.llmodel.model_id,
							  "Generate prompt based on analysis", "Failed to parse prompt tags"):
			new_prompt = _parse_response(response, values.from_anon)

		with ParseError.guard(system_prompt, user_message, response, self.llmodel.model_id,
							  "Generate prompt based on analysis", "Failed to parse prompt tags"):
			dataset.iterations[values.current_iteration].analysis_summary = extract_tag(response, 'last_iteration_summary').strip()
			dataset.iterations[values.current_iteration].prompt_changes = extract_tag(response, 'prompt_changes').strip()

		return new_prompt

	def generate_first_prompt(self) -> Prompt:
		user_message = "Generate initial prompt."
		response = self.llmodel.request(self.first_system_prompt, user_message)
		with ParseError.guard(self.first_system_prompt, user_message, response, self.llmodel.model_id,
							  "Generate first prompt", "Failed to parse prompt tags"):
			new_prompt = _parse_response(response, {})
		return new_prompt

	@staticmethod
	def get_all_models_context(dataset: ResultDataset, models_anon: dict):
		context_list = []
		for iteration_info in dataset.iterations:
			if iteration_info.analysis_summary is not None:
				context_list.append(get_summary_context(dataset, models_anon, iteration_info.iteration))
			else:
				context_list.append(get_full_context(dataset, models_anon, iteration_info.iteration))

		return '\n\n'.join(context_list)


