import os
import requests
import json
import re
import time
import numpy as np

from mw import make
from prompt_cot import demo_prompt, system_prompt, new_task_prompt, interact_prompt, cot_prompt

API_KEY = os.environ.get("OPENAI_API_KEY")
model_choice = "gpt-4-32k"
API_ENDPOINT = (
    f"https://gcrgpt4aoai5c.openai.azure.com/openai/deployments/{model_choice}"
    f"/chat/completions?api-version=2023-03-15-preview"
)
headers = {'Content-Type': 'application/json', 'api-key': API_KEY}
max_wait_gpt4_time = 40


def get_first_input(obs):
    """Build the prompt for the very first step of an episode."""
    obs = np.array(obs)
    new_task_prompt_new = new_task_prompt.format(observation=obs)
    input_data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": demo_prompt},
            {"role": "user", "content": new_task_prompt_new},
        ],
        "max_tokens": 500,
        "temperature": 0,
    }
    return input_data


def get_input(obs, action, history_obs, count, pred_obs):
    """Build the prompt for subsequent steps, alternating between action and CoT turns."""
    obs = np.array(obs)
    action = np.array(action)
    input_data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": demo_prompt},
            {"role": "user", "content": new_task_prompt},
        ],
        "max_tokens": 500,
        "temperature": 0.7,
    }
    if count % 2 == 0:
        turn_prompt = interact_prompt.format(
            previous_history=history_obs,
            current_observation=obs,
            previous_action=action,
        )
    else:
        turn_prompt = cot_prompt.format(observation=obs)
    input_data["messages"].append({"role": "user", "content": turn_prompt})
    return input_data


def parse_action(llm_output_str):
    """Extract the 4-dim action vector from the LLM response."""
    raw = llm_output_str.split('The predicted current action is [')[1].split('],')[0]
    return np.array([float(v.strip()) for v in raw.split(',') if v.strip()])


def parse_predicted_observation(llm_output_str):
    """Extract the predicted next observation from the LLM response."""
    raw = llm_output_str.split('The predicted next observation is [')[1].split('].')[0]
    return np.array([float(v.replace(']', '').strip()) for v in raw.split(',') if v.strip()])


def call_api_with_retry(input_data):
    """Post to the GPT-4 endpoint, retrying on rate-limit errors."""
    while True:
        try:
            response = requests.post(API_ENDPOINT, json=input_data, headers=headers, timeout=30)
            result = response.json()
        except Exception:
            continue
        if 'error' in result:
            message = result['error']['message']
            sleep_time = int(re.findall(r'Please retry after (\w+) second', message)[0])
            time.sleep(min(sleep_time, max_wait_gpt4_time) + 1.0)
        else:
            return result


if __name__ == "__main__":
    train_env = make(name='door-open', frame_stack=3, action_repeat=2, seed=1,
                     train=True, device_id=-1)
    time_step = train_env.reset()

    count = 0
    history_obs = []
    action = np.zeros(4)
    predicted_observation = None

    save_path = os.path.join(os.path.dirname(__file__), 'output', 'run.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    while not time_step.last() and time_step['success'] != 1 and count <= 100:
        observation = np.array(time_step.observation)

        if count == 0:
            input_data = get_first_input(obs=observation)
        else:
            input_data = get_input(
                obs=observation,
                action=action,
                history_obs=history_obs,
                count=count,
                pred_obs=predicted_observation,
            )

        llm_output = call_api_with_retry(input_data)

        with open(save_path, "w") as f:
            json.dump(llm_output, f)

        if count % 2 == 0:
            action_str = llm_output['choices'][0]['message']['content']
            action = parse_action(action_str)
            predicted_observation = parse_predicted_observation(action_str)
            history_obs.append(observation)
            history_obs = history_obs[-10:]
            time_step = train_env.step(action)

        count += 1
