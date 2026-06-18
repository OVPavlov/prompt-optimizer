# Prompt Optimizer

Uses an AI model's domain knowledge to optimize prompts and measure output quality.

1. Choose a capable model with enough domain knowledge for quality control.
2. Provide a task description and representative input data.
3. Prompt Optimizer generates a task-specific rating schema.
4. It runs one or more executor models on sampled input data.
5. It analyzes the results and rewrites the prompt to improve measured quality.

The optimizer stores each iteration on disk. An interrupted or completed experiment can be loaded and continued later.

## API key configuration

Pass an API key directly:
```python
from prompt_optimizer import LLMClient
client = LLMClient.openrouter(api_key="...")
```
Or set one of these environment variables, then create the corresponding client:
```python
client = LLMClient.openrouter()
# client = LLMClient.openai()
```

## Basic usage

```python
from prompt_optimizer import PromptOptimizer, LLMClient

client = LLMClient.openrouter()

task_description = """
Describe the executor's task, required output, and correctness constraints.
The generated system prompt may contain {per_model_instructions}.
"""

input_data = [
    {"input": "representative item 1"},
    {"input": "representative item 2"},
    {"input": "representative item 3"},
]

executor_models = [
    "provider/executor-model-a",
    "provider/executor-model-b",
]

optimizer = PromptOptimizer.create_standard(
    directory="experiments/example",
    task_description=task_description,
    all_data=input_data,
    client=client,
    models=executor_models,
    main_model="provider/optimizer-model",
)

optimizer.run_experiments(iterations=3, num_data=3)
```

Use the constructor when you need explicit control over the meta-prompt, analysis, and prompt-generation models:

```python
from prompt_optimizer import PromptOptimizer, LLMClient

client = LLMClient.openrouter()

meta_prompt_model = client.get_reasoning_model(
    "provider/schema-model",
    effort="high",
)
analysis_model = client.get_reasoning_model(
    "provider/analysis-model",
    effort="low",
)
prompt_model = client.get_reasoning_model(
    "provider/generator-model",
    effort="medium",
)

optimizer = PromptOptimizer(
    directory="experiments/example",
    task_description=task_description,
    all_data=input_data,
    client=client,
    meta_prompt_model=meta_prompt_model,
    analysis_model=analysis_model,
    prompt_model=prompt_model,
    models=executor_models,
)
```

Set `use_task_as_first_prompt=True` to use the task description directly instead of generating an initial prompt.

## Resume an experiment

```python
from prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer.load("experiments/example")
optimizer.run_experiments(iterations=2, num_data=8)
```

`load()` reconstructs the client and models from saved parameters. The matching API key must still be available through the environment.

## Inspect results

Display ratings for all executor models:

```python
optimizer.display_all_models()
```

Display one model:

```python
optimizer.display_model("provider/executor-model-a")
```

Print the input/output pairs from one iteration:

```python
optimizer.print_outputs("provider/executor-model-a", iteration=0)
```

Access structured iteration data:

```python
iteration = optimizer.dataset.get_i(0)
model_result = optimizer.dataset.get_mr("provider/executor-model-a", 0)

print(iteration.prompt)
print(model_result.analysis)
print(model_result.rating)
```

Inspect generated prompt history:

```python
context = optimizer.prompt_generator.get_all_models_context(
    optimizer.dataset,
    optimizer.values.to_anon,
)
print(context)
```

## Cost, token, and latency statistics

```python
optimizer.display_stats()
optimizer.get_stats_avg()
optimizer.get_stats_sum()
optimizer.get_cost()
optimizer.get_latencies()
```

Statistics are persisted under the experiment directory and restored by `PromptOptimizer.load()`.

## Select executor models

Remove models from subsequent iterations:

```python
optimizer.values.remove_models(
    ["provider/executor-model-b"],
    reshuffle=True,
)
```

Replace an executor model configuration:

```python
optimizer.ll_models["provider/executor-model-a"] = (
    client.get_reasoning_model(
        "provider/executor-model-a",
        effort="medium",
        stats_path=optimizer.llm_stats_path,
        stats_key="executor-model-a-medium",
    )
)
```
