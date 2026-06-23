import json
from pathlib import Path
import dataclasses
import typing
from dataclasses import asdict
from .DataTypes import DataOutput, Prompt, ModelResult, IterationInfo, ResultDataset, ExperimentValues



def load_json(path:Path):
	if path.exists():
		return json.loads(path.read_text())
	else:
		return None

def from_dict(cls, data: dict):
	hints = typing.get_type_hints(cls)
	kwargs = {}
	for field in dataclasses.fields(cls):
		if field.name not in data:
			continue
		value = data[field.name]
		ftype = hints[field.name]
		if dataclasses.is_dataclass(ftype) and isinstance(value, dict):
			kwargs[field.name] = from_dict(ftype, value)
		elif hasattr(ftype, '__origin__') and ftype.__origin__ is list:
			item_type = ftype.__args__[0]
			if dataclasses.is_dataclass(item_type) and isinstance(value, list):
				kwargs[field.name] = [from_dict(item_type, item) if isinstance(item, dict) else item for item in value]
			else:
				kwargs[field.name] = value
		else:
			kwargs[field.name] = value
	return cls(**kwargs)


def load_class(cls, path:Path):
	return from_dict(cls, load_json(path))



class Logger:
	def __init__(self, log_dir: str):
		self.root = Path(log_dir)
		self.root.mkdir(parents=True, exist_ok=True)

	def get_iteration_dir(self, iteration: int) -> Path:
		return self.root / f"iter_{iteration:03d}"

	def write(self, iteration: int, filename: str, data: dict|list):
		d = self.get_iteration_dir(iteration)
		d.mkdir(exist_ok=True)
		(d / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False))

	def write_sd(self, iteration: int, subdir: str, filename: str, data: dict):
		d = self.get_iteration_dir(iteration) / subdir
		d.mkdir(exist_ok=True)
		(d / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False))

	def _mrs(self, dataset: ResultDataset, iteration: int) -> list[ModelResult]:
		return [dataset.get_mr(m, iteration) for m in dataset.get_i(iteration).models]

	def log_info(self, info: IterationInfo):
		self.write(info.iteration, "info.json", asdict(info))

	def log_response(self, iteration: int, model:str, data_id:int, output:DataOutput):
		model = model.replace('/','_').replace('\\','_').replace('-','_')
		self.write_sd(iteration, 'responses', f"response{model}_{data_id}.json", asdict(output))

	def log_results(self, dataset: ResultDataset, iteration: int):
		self.write(iteration, "results.json", [asdict(mr) for mr in self._mrs(dataset, iteration)])

	def log_analysis(self, dataset: ResultDataset, iteration: int):
		self.write(iteration, "analysis.json", {
			mr.model: {"analysis": mr.analysis, "rating": mr.rating}
			for mr in self._mrs(dataset, iteration)})

	def log_new_prompt(self, iteration: int, prompt: Prompt):
		self.write(iteration, "new_prompt.json", asdict(prompt))

	def load_iterations(self) -> list[IterationInfo]:
		iter_dirs = sorted(self.root.glob("iter_*"))
		iteration_infos:list[IterationInfo] = []
		for dir in iter_dirs:
			iteration_infos.append(load_class(IterationInfo, dir / "info.json"))
		return iteration_infos

	def load_results(self, iteration: int) -> list[ModelResult]:
		dir = self.get_iteration_dir(iteration)
		res = load_json(dir / "results.json")
		anal = load_json(dir / "analysis.json")
		results:list[ModelResult] = [from_dict(ModelResult, mr) for mr in res]
		if anal is not None:
			for res in results:
				an = anal[res.model]
				res.analysis = an["analysis"]
				res.rating = an["rating"]
		return results

	def has_info(self, iteration: int) -> bool:
		return (self.get_iteration_dir(iteration) / "info.json").exists()

	def has_results(self, iteration: int) -> bool:
		return (self.get_iteration_dir(iteration) / "results.json").exists()

	def has_analysis(self, iteration: int) -> bool:
		return (self.get_iteration_dir(iteration) / "analysis.json").exists()

	def has_new_prompt(self, iteration: int) -> bool:
		return (self.get_iteration_dir(iteration) / "new_prompt.json").exists()

	def load_new_prompt(self, iteration: int) -> Prompt:
		dir = self.get_iteration_dir(iteration)
		return load_class(Prompt, dir / "new_prompt.json")

	def load_dataset(self) -> ResultDataset:
		iterations = self.load_iterations()
		if len(iterations) == 0:
			return None

		dataset = ResultDataset(iterations[0].models)
		for iteration in iterations:
			dataset.add_i(iteration)
			results = self.load_results(iteration.iteration)
			for mr in results:
				dataset.add_mr(mr)

		return dataset
