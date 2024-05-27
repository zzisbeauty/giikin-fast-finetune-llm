import os, sys

proPtah = os.getcwd()
sys.path.append(proPtah)

from giikin_functions import *


def get_prompt_chn2eng(detectLang, targetLang):
    # detectLang = '简体中文'
    # targetLang = '英文'
    # demo trans
    # transTask = '紫外线由UVA、UVB、UVC组成，其中UVA就是"皮肤杀手"'
    prompt = f"""你是一位优秀的翻译专家，擅长从{detectLang}到{targetLang}的专业翻译，能够信雅达的翻译出{targetLang}的生动和准确。接下来请准确将如下电商文案：
    """
    return prompt


all_chn2tar_data = []
# YL Data
ylData = '../read_and_clean_yl_trans_data/data_process_arc1data_4_v1transmodel/lang_type_trans_data_ori_tar_more_clean.pick'
with open(ylData, 'rb') as f:
    yl_data = pickle.load(f)
    for k, v in yl_data.items():
        tarLang = ''
        if k == 'TH':
            tarLang = '泰语'
        if k == 'JP':
            tarLang = '日语'
        if k == 'KR':
            tarLang = '韩语'
        if k == 'EN':
            tarLang = '英语'
        for _, vv in v.items():
            for i in vv:
                prompt = get_prompt_chn2eng('简体中文', tarLang)
                all_chn2tar_data.append({'instruction': prompt, 'input': i[0], 'output': i[1]})
                # all_chn2th_data_to_change_chn_2_eng.append({
                #     '简体中文': i[0],
                #     '简体中文-英文': '',
                #     '泰语': i[1],
                # })

with open('all_chn2tar_data.json', 'w', encoding='utf-8') as f:
    json.dump(all_chn2tar_data, f, ensure_ascii=False)
