import os
import sys
from evaluator import Evaluator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chatdev.rsde_bench import softwares

PROMPT_TEMPLATE = """Complete the following task:
{task}
"""

if __name__ == "__main__":
    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model = "text-embedding-v4"
    evaluator = Evaluator(api_key, base_url, embedding_model)

    data_root = "/home/quzhan/quzhan/data/llm/rSDE-bench_4.1-mini"
    requirements = []
    root_dir = []
    for num_sample in range(5):
        sample_dir = os.path.join(data_root, f"gpt-4.1-mini-{num_sample}")
        # sample_dir = os.path.join(data_root, f"ChatDev-{num_sample}")
        for task in os.listdir(sample_dir):
            root_dir.append(os.path.join(sample_dir, task))
            # rSDE-Bench
            for problem in softwares:
                if problem.get("name") == task:
                    prompt = PROMPT_TEMPLATE.format(task=problem["task"])
                    requirements.append(prompt)
                    break

    results = evaluator.evaluate(requirements, root_dir)
    print(f"Completeness Score: {results[0]:.4f}")
    print(f"Executability Score: {results[1]:.4f}")
    print(f"Consistency Score: {results[2]:.4f}")
    print(f"Quality Score: {results[3]:.4f}")

# Completeness Score: 0.9400
# Executability Score: 1.0000
# Consistency Score: 0.8334
# Quality Score: 0.7834

# Completeness Score: 0.9600
# Executability Score: 1.0000
# Consistency Score: 0.8550
# Quality Score: 0.8208