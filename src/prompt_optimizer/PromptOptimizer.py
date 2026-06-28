import json
import difflib
import shutil
import subprocess
from random import sample
from .LLMClient import LLMClient, LLModel, RequestStats
import pandas as pd
from pathlib import Path
from dataclasses import asdict
from .DataTypes import DataOutput, Prompt, ModelResult, IterationInfo, ResultDataset, ExperimentValues, ParallelRequest
from .Logger import Logger
from .MetaPrompt import MetaPrompt
from .ResultAnalyzer import ResultAnalyzer
from .PromptGenerator import PromptGenerator
from .PromptCommon import get_ratings, extract_all


class PromptOptimizer:
	def __init__(self, directory:str, task_description:str, all_data:list,
				 client:LLMClient, meta_prompt_model:LLModel, analysis_models:LLModel|list[LLModel], prompt_model:LLModel, models:list[str],
				 analysis_max_data_items:int|None=None, use_task_as_first_prompt:bool=False, save_params:bool=True):
		self.client = client
		self.dir = Path(directory)
		self.logger = Logger(f"{directory}/log")
		self.task_description = task_description
		self.all_data = all_data
		self.llm_stats_path = self.logger.root / "llm_stats.json"

		self.ll_models: dict[str, LLModel] = {m:client.get_LLmodel(m, stats_path=self.llm_stats_path, stats_key=m) for m in models}

		analysis_models = analysis_models if isinstance(analysis_models, list) else [analysis_models]

		meta_prompt_model.load_stats(self.llm_stats_path, "meta_prompt_model")
		for i, am in enumerate(analysis_models):
			am.load_stats(self.llm_stats_path, "analysis_model" if len(analysis_models) == 1 else f"analysis_model_{i}")
		prompt_model.load_stats(self.llm_stats_path, "prompt_model")

		self.meta_prompt = MetaPrompt(directory, meta_prompt_model, self.task_description)
		self.meta_prompt.generate_rating_schema() # load or generate

		self.result_analyzer = ResultAnalyzer(self.meta_prompt.analysis_system_message, None, analysis_models, analysis_max_data_items)
		self.prompt_generator = PromptGenerator(self.meta_prompt, "{context}", prompt_model)

		self.values = ExperimentValues(models)
		if self.dir.exists() and len(list(self.logger.root.glob("iter_*"))) > 0:
			self.dataset = self.logger.load_dataset()
			self._restore_values_from_log()
		else:
			self.dataset = ResultDataset(models)
			self.values.prompt = self.get_first_prompt(use_task_as_first_prompt=use_task_as_first_prompt) # load or generate

		if save_params:
			self._save_params()

	def _set_iteration_state(self, info: IterationInfo):
		self.values.current_iteration = info.iteration
		self.values.prompt = info.prompt
		self.values.current_models = info.models
		self.values.data_list = info.data

	def _restore_values_from_log(self):
		last_info = self.dataset.iterations[-1]
		self._set_iteration_state(last_info)
		if self.logger.has_new_prompt(last_info.iteration):
			self.values.prompt = self.logger.load_new_prompt(last_info.iteration)
			self.values.current_iteration = last_info.iteration + 1
			self.values.data_list = []

	def _save_params(self):
		params_path = self.dir / "params.json"
		params_dict = {
			'all_data': self.all_data,
			'client': self.client.base_url,
			'meta_prompt_model': self.meta_prompt.llmodel.to_dict(self.dir),
			'analysis_models': [m.to_dict(self.dir) for m in self.result_analyzer.llmodels],
			'analysis_max_data_items': self.result_analyzer.max_data_items,
			'prompt_model': self.prompt_generator.llmodel.to_dict(self.dir),
			'models': {k:v.to_dict(self.dir) for k, v in self.ll_models.items()}
		}
		params_path.write_text(json.dumps(params_dict, indent=2, ensure_ascii=False))

	@staticmethod
	def load(directory:str):
		dir = Path(directory)
		task_description = (dir / "task_description.txt").read_text()
		params = json.loads((dir / "params.json").read_text())
		all_data = params["all_data"]
		client = LLMClient.from_base_url(params["client"])
		models_dict = params["models"]

		prompt_optimizer = PromptOptimizer(directory, task_description, all_data, client,
							meta_prompt_model=LLModel.from_dict(params["meta_prompt_model"], dir),
							analysis_models=[LLModel.from_dict(m, dir) for m in params["analysis_models"]],
							analysis_max_data_items=params.get("analysis_max_data_items"),
							prompt_model=LLModel.from_dict(params["prompt_model"], dir),
							models=list(models_dict.keys()), save_params=False)

		prompt_optimizer.ll_models = {k:LLModel.from_dict(v, dir) for k, v in models_dict.items()}
		for llmodel in prompt_optimizer.ll_models.values():
			llmodel.load_stats(llmodel.stats_path, llmodel.stats_key)

		return prompt_optimizer

	@staticmethod
	def create_standard(directory:str, task_description:str, all_data:list, client:LLMClient, models:list[str], main_model:str):
		return PromptOptimizer(directory, task_description, all_data, client,
					 meta_prompt_model=client.get_reasoning_model(main_model, effort="high"),
					 analysis_models=client.get_reasoning_model(main_model, effort="low"),
					 prompt_model=client.get_reasoning_model(main_model, effort="medium"),
					 models=models)

	def get_first_prompt(self, use_task_as_first_prompt:bool=False) -> Prompt:
		first_prompt_path = self.dir / "first_prompt.json"

		if first_prompt_path.exists():
			data = json.loads(first_prompt_path.read_text())
			return Prompt(**data)
		else:
			prompt = Prompt(self.task_description,None,None) if use_task_as_first_prompt else self.prompt_generator.generate_first_prompt()
			first_prompt_path.write_text(json.dumps(asdict(prompt), indent=2, ensure_ascii=False))
			return prompt

	def _request(self, system:str, user:str, model:str) -> str:
		if model not in self.ll_models:
			self.ll_models[model] = self.client.get_LLmodel(model)
		llmodel = self.ll_models[model]

		return llmodel.request(system,user)


	def generate_results_parallel(self):
		info = self.dataset.get_i(self.values.current_iteration)
		if info is None:
			info = IterationInfo(self.values.current_iteration, self.values.prompt, self.values.current_models, self.values.data_list)
			self.dataset.add_i(info)
		else:
			self._set_iteration_state(info)
		self.logger.log_info(info)

		for model in self.values.current_models:
			if model not in self.ll_models:
				self.ll_models[model] = self.client.get_LLmodel(model)

		requests = []
		for model in self.values.current_models:
			mr = ModelResult(model, self.values.current_iteration, self.values.prompt.get_for(model), [])
			self.dataset.add_mr(mr)
			for data_id, data in enumerate(self.values.data_list):
				requests.append(ParallelRequest(self.ll_models[model],
					self.values.prompt.system_message_for(model), str(data), (mr, data_id, data)))

		ParallelRequest.run(requests)

		for req in requests:
			mr, data_id, data = req.tag
			output = DataOutput(data, req.output)
			mr.results.append(output)
			# self.logger.log_response(self.values.current_iteration, mr.model, data_id, output)

		self.logger.log_results(self.dataset, self.values.current_iteration)


	def run_experiments(self, iterations:int, num_data:int):
		for i in range(iterations):
			print(f"Iteration {i}/{iterations}")

			info = self.dataset.get_i(self.values.current_iteration)
			if info is not None:
				self._set_iteration_state(info)
			else:
				self.values.data_list = sample(self.all_data, num_data)

			if not self.logger.has_results(self.values.current_iteration):
				print(f"\tGenerate Results")
				self.generate_results_parallel()

			if not self.logger.has_analysis(self.values.current_iteration):
				print(f"\tAnalyse Results")
				self.result_analyzer.generate_analysis_parallel(self.dataset, self.values.current_models, self.values.current_iteration)
				self.logger.log_analysis(self.dataset, self.values.current_iteration)

			if not self.logger.has_new_prompt(self.values.current_iteration):
				print(f"\tGenerating Prompt")
				new_prompt = self.prompt_generator.generate_prompt(self.dataset, self.values)
				self.logger.log_new_prompt(self.values.current_iteration, new_prompt)
				self.values.prompt = new_prompt
				self.logger.log_info(self.dataset.get_i(self.values.current_iteration))

			self.values.current_iteration += 1



	def display_all_models(self, display_pass=True, display_rating=True):
		both = display_pass and display_rating
		columns = {}
		if self.dataset.iterations is None or len(self.dataset.iterations) == 0:
			return
		bool_num = 0
		for m in self.dataset.iterations[-1].models:
			df = pd.DataFrame([r.rating for r in self.dataset.results[m]])
			name = self._short_name(m)
			bool_num = max(bool_num, len(df.select_dtypes(include='bool').columns))
			if display_pass:
				columns[f'{name}_pass' if both else name] = df.select_dtypes(include='bool').sum(axis=1)
			if display_rating:
				columns[f'{name}_rating' if both else name] = df.select_dtypes(include='number').mean(axis=1)

		df, cols = pd.DataFrame(columns), list(columns)
		style = df.style
		if display_pass:
			style = style.bar(subset=[c for c in cols if c.endswith('_pass')] or cols, cmap='RdYlGn', vmin=0, vmax=bool_num)
		if display_rating:
			rating_subset = [c for c in cols if c.endswith('_rating')] or cols
			style = style.bar(subset=rating_subset, cmap='RdYlGn', vmin=0, vmax=1).format("{:.0%}", subset=rating_subset)
		display(style)

	def display_model(self, model: str):
		if self.dataset.iterations is None or len(self.dataset.iterations) == 0:
			return
		if self.dataset.results is None or model not in self.dataset.results:
			return
		df = pd.DataFrame([r.rating for r in self.dataset.results[model]])
		floats = df.select_dtypes(include='number').columns
		bools = df.select_dtypes(include='bool').columns

		def color_bool(val):
			return 'background-color: #263' if val else 'background-color: #621'

		display(df.style
				.bar(cmap='RdYlGn', vmin=0, vmax=1)
				.format("{:.0%}", subset=floats)
				.map(color_bool, subset=bools))

	def display_all_models_uni(self):
		columns = {}
		if self.dataset.iterations is None or len(self.dataset.iterations) == 0:
			return
		for m in self.dataset.iterations[-1].models:
			df = pd.DataFrame([r.rating for r in self.dataset.results[m]])
			name = self._short_name(m)
			columns[name] = df.mean(axis=1).astype(float)
		display(pd.DataFrame(columns).style.bar(subset=list(columns), cmap='RdYlGn', vmin=0, vmax=1).format("{:.0%}"))

	@staticmethod
	def _schema_fields(section:str|None) -> list[tuple[str, str, str]]:
		return [(t.attrs.get("name", ""), t.attrs.get("type", ""), t.body.strip())
				for t in extract_all(section or "", "field")]

	def _schema_by_field(self) -> dict[str, dict]:
		mp = self.meta_prompt
		schema = {n: (t, d) for n, t, d in self._schema_fields(mp.schema_design)}
		analysis = {n: d for n, _, d in self._schema_fields(mp.analysis_guidance)}
		generation = {n: d for n, _, d in self._schema_fields(mp.generation_guidance)}
		return {n: {"type": t,
					"schema_design": d,
					"analysis_guidance": analysis.get(n, ""),
					"generation_guidance": generation.get(n, "")}
				for n, (t, d) in schema.items()}

	def display_schema_table(self, is_editable:bool=False):
		data = self._schema_by_field()

		if not is_editable:
			df = pd.DataFrame.from_dict(data, orient="index").rename_axis("field")

			def color_type(v):
				bg = '#284' if v == 'bool' else '#c82'
				return f'background-color:{bg};color:white;font-weight:600;text-align:center;'

			display(df.style
					.map(color_type, subset=["type"])
					.set_properties(**{'text-align': 'left', 'white-space': 'normal', 'vertical-align': 'top'})
					.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}]))
			return

		import ipywidgets as W
		mp = self.meta_prompt
		sections = [k for k in next(iter(data.values()), {}) if k != "type"]
		areas, children = {}, []

		children.append(W.HTML("<b>field</b>"))
		children += [W.HTML(f"<b>{s}</b>") for s in sections]

		for name, info in data.items():
			color = '#284' if info["type"] == 'bool' else '#c82'
			children.append(W.HTML(
				f'<code>{name}</code><br>'
				f'<span style="color:{color};font-weight:600;">{info["type"]}</span>'))
			areas[name] = {s: W.Textarea(value=info[s], layout=W.Layout(width="100%", height="120px")) for s in sections}
			children += [areas[name][s] for s in sections]

		grid = W.GridBox(children, layout=W.Layout(grid_template_columns="130px 1fr 1fr 1fr", grid_gap="4px", width="100%"))
		out = W.Output()
		btn = W.Button(description="Save", button_style="success")

		def save(_):
			parts = {}
			for s in sections:
				blocks = []
				for n, info in data.items():
					type_attr = ' type="{}"'.format(info["type"]) if s == "schema_design" else ""
					blocks.append('<field name="{}"{}>\n{}\n</field>'.format(
						n, type_attr, areas[n][s].value.strip()))
				parts[s] = "\n".join(blocks)
			mp.write_file("rating_schema_source",
				"```xml\n"
				f"<schema_design>\n{parts['schema_design']}\n</schema_design>\n\n"
				f"<analysis_guidance>\n{parts['analysis_guidance']}\n</analysis_guidance>\n\n"
				f"<generation_guidance>\n{parts['generation_guidance']}\n</generation_guidance>\n```")
			mp.generate_rating_schema()
			self.result_analyzer.system_prompt = mp.analysis_system_message
			with out:
				out.clear_output()
				print("Saved →", mp.dir / "rating_schema_source.txt")

		btn.on_click(save)
		display(W.VBox([grid, btn, out]))

	def display_schema_cards(self, all_opened:bool=False):
		from IPython.display import HTML
		cards = []
		for i, (name, info) in enumerate(self._schema_by_field().items()):
			accent = '#284' if info["type"] == 'bool' else '#c82'
			is_open = " open" if all_opened or (i == 0) else ""
			body = "".join(
				f'<div style="margin-top:6px;"><span style="color:#666;font-weight:800;'
				f'font-size:.9em;letter-spacing:.04em;">{s.upper().replace("_", " ")}</span>'
				f'<div style="color:#bbb;font-size:.9em">{text}</div></div>'
				for s, text in info.items() if s != "type")
			cards.append(
				f'<details{is_open} style="border-left:4px solid {accent};margin:8px 0;'
				f'background:rgba(127,127,127,.08);border-radius:4px;">'
				f'<summary style="cursor:pointer;padding:8px 14px;">'
				f'<code style="font-size:1.05em;">{name}</code>'
				f'<span style="color:{accent};font-weight:600;"> &middot; {info["type"] or "?"}</span>'
				f'</summary>'
				f'<div style="padding:0 14px 8px;">{body}</div></details>')
		display(HTML("".join(cards)))

	def print_prompt_history(self, model:str|None=None, show_diff:bool=True, show_stated_changes:bool=True,
							 iterations:int|range|None=None):
		if model is not None and model not in self.dataset.results:
			raise ValueError(f"Unknown model: {model}")
		if iterations is not None and (isinstance(iterations, bool) or not isinstance(iterations, (int, range))):
			raise TypeError("iterations must be an int, range, or None")

		old_prompt = None
		old_rating = None
		previous_changes = None
		displayed = 0

		for info in self.dataset.iterations:
			if model is None:
				parts = [info.prompt.system.strip("\n")]
				if isinstance(info.prompt.instructions, dict):
					parts += [f"[{name}]\n{text.strip(chr(10))}"
							  for name, text in info.prompt.instructions.items() if text is not None]
				elif info.prompt.instructions:
					parts.append(info.prompt.instructions.strip("\n"))
				rating = get_ratings(self.dataset, info.iteration)
			else:
				instructions = info.prompt.instructions
				instructions = instructions.get(model) if isinstance(instructions, dict) else instructions
				parts = [info.prompt.system.replace(
					"{per_model_instructions}", instructions or "").strip("\n")]
				model_result = self.dataset.get_mr(model, info.iteration)
				rating = model_result.rating if model_result is not None and model_result.rating is not None else {}

			if info.prompt.user_message is not None:
				parts.append(info.prompt.user_message.strip("\n"))
			prompt = "\n\n".join(parts)

			if iterations is None:
				selected = True
			elif isinstance(iterations, int):
				selected = info.iteration == iterations
			else:
				selected = info.iteration in iterations

			if not selected:
				old_prompt = prompt
				old_rating = rating
				previous_changes = info.prompt_changes
				continue

			displayed += 1
			print(f"\033[1;38;5;208m{'=' * 25} ITERATION {info.iteration} {'=' * 25}\033[0m")
			if not show_diff or old_prompt is None:
				print(prompt)
			else:
				old_lines = old_prompt.splitlines()
				new_lines = prompt.splitlines()
				diff = list(difflib.unified_diff(
					old_lines, new_lines, fromfile="old", tofile="new",
					lineterm="", n=max(len(old_lines), len(new_lines))))
				if not diff:
					print(prompt)
				else:
					delta = shutil.which("delta")
					try:
						rendered = subprocess.run([
							delta, "--no-gitconfig", "--paging=never", "--file-style=omit",
							"--hunk-header-style=omit", "--syntax-theme=none",
							"--keep-plus-minus-markers", "--width=variable",
							"--minus-style=red", "--minus-emph-style=red bold reverse",
							"--plus-style=green", "--plus-emph-style=green bold reverse",
							"--zero-style=normal", "--max-line-distance=0.8",
						], input="\n".join(diff) + "\n", text=True,
							capture_output=True, check=True).stdout if delta else None
					except (OSError, subprocess.CalledProcessError):
						rendered = None

					if rendered:
						print(rendered.lstrip("\n"), end="")
					else:
						for line in diff[3:]:
							color = "\033[31m" if line.startswith("-") else \
									"\033[32m" if line.startswith("+") else ""
							print(f"{color}{line}\033[0m" if color else line)

			if show_stated_changes:
				print("\033[1mPrompt changes\033[0m")
				if old_prompt is None:
					print("Initial prompt")
				else:
					print(previous_changes if previous_changes else "Not found")

			print("\033[1mRatings\033[0m")
			if not rating:
				print("Not found")
			for name, value in rating.items():
				if old_rating is None or name not in old_rating:
					print(f"{name}: {value:.0%}")
					continue
				change = value - old_rating[name]
				color = "\033[32m" if change > 0 else "\033[31m" if change < 0 else ""
				numbers = f"{value:.0%} ({change:+.0%})"
				print(f"{name}: {color}{numbers}\033[0m" if color else f"{name}: {numbers}")
			print()

			old_prompt = prompt
			old_rating = rating
			previous_changes = info.prompt_changes

		if iterations is not None and displayed == 0:
			raise ValueError(f"No matching iterations found for {iterations}")

	@staticmethod
	def _short_name(model_id:str) -> str:
		return model_id[model_id.index('/') + 1:] if '/' in model_id else model_id

	def _components(self) -> tuple[list[tuple[str, LLModel]], list[tuple[str, LLModel]]]:
		pipeline = [
			("meta_prompt", self.meta_prompt.llmodel),
			("prompt_generator", self.prompt_generator.llmodel),
		]
		pipeline += [(f"analysis_{self._short_name(m.model_id)}", m)
					 for m in self.result_analyzer.llmodels]
		test = [(f"TEST_{self._short_name(k)}", v) for k, v in self.ll_models.items()]
		return pipeline, test

	def get_cost(self):
		pipeline, test = self._components()
		cost_dict = {label: model.get_stats_avg().cost for label, model in pipeline + test}
		cost_dict['TEST_TOTAL'] = sum(cost_dict[label] for label, _ in test)
		cost_dict['Total Cost'] = sum(cost_dict[label] for label, _ in pipeline + test)
		return cost_dict

	def get_latencies(self):
		pipeline, test = self._components()
		return {label: model.avg_latency for label, model in pipeline + test}

	def display_stats(self):
		costs = self.get_cost()
		latencies = self.get_latencies()

		df = pd.DataFrame({
			"Cost ($)": costs,
			"Avg Latency (s)": latencies,
		})
		df.index.name = "Component"
		display(df)

	def _aggregate_stats(self, aggregator:callable) -> dict[str, RequestStats]:
		pipeline, test = self._components()
		stat_lists = {label: model.stats_list for label, model in pipeline + test}

		d = {k: aggregator(v) for k, v in stat_lists.items()}
		total = aggregator(list(d.values()))
		d["TEST_TOTAL"] = aggregator([aggregator(model.stats_list) for _, model in test])
		d["TOTAL"] = total
		d = {k: v for k, v in d.items() if v is not None}
		for k, v in d.items():
			v.model = k
		return d

	def get_stats_avg(self):
		stats = self._aggregate_stats(RequestStats.avg)
		df = pd.DataFrame(stats.values())
		df.drop(columns=['time', 'finish_reason'], inplace=True)
		df.set_index('model', inplace=True)
		return df

	def get_stats_sum(self):
		stats = self._aggregate_stats(RequestStats.sum)
		df = pd.DataFrame(stats.values())
		df.drop(columns=['time', 'finish_reason'], inplace=True)
		df.set_index('model', inplace=True)
		return df

	def print_outputs(self, model:str, iteration:int):
		mr = self.dataset.get_mr(model, iteration)
		if mr is None:
			raise Exception(f"No {model} in iteration {iteration}")
		for i, res in enumerate(mr.results):
			print('='*25, f'\t\tdata[{i}]\t\t', '='*25)
			print(res.data)
			print('-'*70)
			print(res.output)
			print('-'*70)
			print('\n')
