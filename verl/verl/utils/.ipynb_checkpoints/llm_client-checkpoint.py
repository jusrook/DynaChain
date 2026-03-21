from openai import OpenAI
import json

client = OpenAI(
    api_key="sk-39143a0e12564d80867ac2b0badbb65b",  # 如果您没有配置环境变量，请用阿里云百炼API Key将本行替换为：api_key="sk-xxx"
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope SDK的base_url
)

def get_response(input_data,system_message="You are a helpful assistant.",logprobs=False,model="qwen-plus"):
    completion = client.chat.completions.create(
        model=model,  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[{'role': 'system', 'content': f'{system_message}'},
                  {'role': 'user', 'content': f'{input_data}'}],
        logprobs=logprobs,
        top_logprobs=5
        )
    return json.loads(completion.model_dump_json())