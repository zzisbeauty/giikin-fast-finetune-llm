from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel


# hlf-llama3
mode_path = '/home/giikin-fast-finetune-llm/untrackfiles/hfl-llama-3-chinese-8b-instruct'
lora_path = '/home/giikin-fast-finetune-llm/untrackfiles/output-arc1data-v1model-hfl-llama3-knowledge-input-save-lora'
# lora model
from peft import LoraConfig, TaskType, get_peft_model
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式
    r=8, # Lora 秩
    lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1 # Dropout 比例
)
# load all model
tokenizer = AutoTokenizer.from_pretrained(mode_path) # 加载tokenizer
model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="auto",torch_dtype=torch.bfloat16) # 加载模型
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config) # 加载lora权重

def get_prompt_chn2eng(detectLang, targetLang, transTask):
    # detectLang = '简体中文'
    # targetLang = '英文'
    # demo trans
    # transTask = '紫外线由UVA、UVB、UVC组成，其中UVA就是"皮肤杀手"'
    # transTask = '无需刻意，一秒帅气'
    # transTask = '四核强动力冰擎，冰感飓风秒降温'
    prompt = f"""
    你是一位优秀的翻译专家，擅长从{detectLang}到{targetLang}的专业翻译，能够信雅达的翻译出{targetLang}的生动和准确。接下来请准确将如下电商文案：\n
    ```{transTask}```\n
    """
    return prompt

def model_inference(prompt):
    messages = [
        # {"role": "system", "content": "现在你要扮演皇帝身边的女人--甄嬛"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        eos_token_id=tokenizer.encode('<|eot_id|>')[0]
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response



import json, time, concurrent.futures


# 首先将各语言的中文翻译成英文； 再执行后续的训练
def tast_inference_chn2eng(prompt):
    target = prompt['target']
    model_response = model_inference(prompt['chn2eng'])
    return model_response, target

def cut_len_list(list_, n):
    len_cuts = [list_[i:i + n] for i in range(0, len(list_), n)]
    return len_cuts

# ############################################ jp data

with open('/home/giikin-fast-finetune-llm/dataset/giikin-arc1data-4-train-v1-model-knowledge-input/all_chn2jp_data_to_change_chn_2_eng.json','r',encoding='utf-8') as f:
    jp_chn2eng = datas = json.load(f)
    print(len(jp_chn2eng))
    all_jp_chn2eng_data_prompts_for_tain = []
    for data in jp_chn2eng:
        chn = data['简体中文']
        kr = data['日文']
        prompt = get_prompt_chn2eng('简体中文', '英文', chn)
        all_jp_chn2eng_data_prompts_for_tain.append({'chn2eng': prompt, 'target': kr})
    splitSeqList = cut_len_list(all_jp_chn2eng_data_prompts_for_tain, 11013)
    print(len(splitSeqList))

get_jp_chn2eng_result, wrong_count_kr = [], []
with concurrent.futures.ThreadPoolExecutor(max_workers=30) as executor:
    all_todo = []
    for _, all_jp_chn2eng_data_prompts_for_tain in enumerate(splitSeqList):
        for __, prompt in enumerate(all_jp_chn2eng_data_prompts_for_tain):
            future = executor.submit(tast_inference_chn2eng, prompt) # 此方法即时返回
            all_todo.append(future) # 如果不先加到临时任务，直接运行下面这个 result，将会阻塞。 因此执行这行代码，未来执行
            # chn2eng, target = future.result() # 阻塞
    for future in concurrent.futures.as_completed(all_todo):
        model_response, target = future.result()
        print(model_response, target)
        get_jp_chn2eng_result.append(
            {
                'instruction': get_prompt_chn2eng('英文', '日文', model_response),
                'input':'',
                'output': target
            }
        )
        print(len(get_jp_chn2eng_result))

with open('/home/giikin-fast-finetune-llm/LLaMA3/mulity_eng2jp_data.json','w',encoding='utf-8') as f:
    json.dump(get_jp_chn2eng_result,f, ensure_ascii=False)


# ######################################## 韩文


# with open('/home/giikin-fast-finetune-llm/dataset/giikin-arc1data-4-train-v1-model-knowledge-input/all_chn2kr_data_to_change_chn_2_eng.json','r',encoding='utf-8') as f:
#     kr_chn2eng = datas = json.load(f)
#     print(len(kr_chn2eng))
#     all_kr_chn2eng_data_prompts_for_tain = []
#     for data in kr_chn2eng:
#         chn = data['简体中文']
#         kr = data['韩文']
#         prompt = get_prompt_chn2eng('简体中文', '英文', chn)
#         all_kr_chn2eng_data_prompts_for_tain.append({'chn2eng': prompt, 'target': kr})
#     splitSeqList = cut_len_list(all_kr_chn2eng_data_prompts_for_tain, 28304)
#     print(len(splitSeqList))

# get_kr_chn2eng_result, wrong_count_kr = [], []
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     to_do = []
#     for _, all_kr_chn2eng_data_prompts_for_tain in enumerate(splitSeqList):
#         for __, prompt in enumerate(all_kr_chn2eng_data_prompts_for_tain):
#             future = executor.submit(tast_inference_chn2eng, prompt) # 此方法即时返回
#             to_do.append(future)
#     for future in concurrent.futures.as_completed(to_do):
#         model_response, target = future.result()
#         print(model_response, target)
#         get_kr_chn2eng_result.append(
#             {
#                     'instruction': get_prompt_chn2eng('英文', '韩文', model_response),
#                     'input':'',
#                     'output': target
#             }
#         )
#         print(len(get_kr_chn2eng_result))
        
# with open('/home/giikin-fast-finetune-llm/LLaMA3/mulity_eng2kr_data.json','w',encoding='utf-8') as f:
#     json.dump(get_kr_chn2eng_result,f, ensure_ascii=False)

# ##########################################################################

# with open('/home/giikin-fast-finetune-llm/dataset/giikin-arc1data-4-train-v1-model-knowledge-input/all_chn2th_data_to_change_chn_2_eng.json','r',encoding='utf-8') as f:
#     th_chn2eng = json.load(f)

# all_th_chn2eng_data_prompts_for_train = []
# for data in th_chn2eng:
#     chn = data['简体中文']
#     kr = data['泰语']
#     prompt = get_prompt_chn2eng('简体中文', '英文', chn)
#     all_th_chn2eng_data_prompts_for_train.append({'chn2eng': prompt, 'target': kr})

# get_th_chn2eng_result, wrong_count_th = [], []
# with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
#     to_do = []
#     for _, prompt in enumerate(all_th_chn2eng_data_prompts_for_train):
#         future = executor.submit(tast_inference_chn2eng, prompt)
#         to_do.append(future)
#     try:
#         for future in concurrent.futures.as_completed(to_do):  # 并发执行
#             chn2eng, target = future.result()
#             get_th_chn2eng_result.append(
#                 {
#                     'instruction': get_prompt_chn2eng('英文', '泰语', chn2eng),
#                     'input':'',
#                     'output': target
#                 }
#             )
#     except:
#         wrong_count_th.append(1)
#         pass

# # # 最后执行完的结果
# print('th - chn2eng 翻译过程中出错的数量：', len(wrong_count_th))
# print('可以去训练 eng2th 翻译的数据数量：',len(get_th_chn2eng_result))
# print('th chn2eng 模型推理完的数据量和原始的 th 数据条目是否相等：', len(get_th_chn2eng_result) + len(wrong_count_th), "source len th：", len(th_chn2eng))

# with open('/home/giikin-fast-finetune-llm/LLaMA3/mulity_eng2th_data.json','w',encoding='utf-8') as f:
#     json.dump(get_th_chn2eng_result,f, ensure_ascii=False)
