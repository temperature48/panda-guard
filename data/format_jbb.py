# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 11:32
# User      : yu
# Product   : PyCharm
# Project   : jailbreak-pipeline
# File      : format_jbb.py
# explain   :
import json
import os
import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("harmful_behaviors_jailbreakbench.csv")

    ROOT_DIR = 'JailbreakBench/attack-artifacts'

    for root, dirs, files in os.walk(ROOT_DIR):

        if "test-artifact" in root:
            continue

        for file in files:
            if "gpt-4" not in file:
                continue

            prompts_4 = json.load(open(os.path.join(root, file)))["jailbreaks"]
            prompts_3_5 = json.load(open(os.path.join(root, "gpt-3.5-turbo-1106.json")))["jailbreaks"]

            if "vicuna-13b-v1.5.json" in files:
                prompts_vicuna = json.load(open(os.path.join(root, "vicuna-13b-v1.5.json")))["jailbreaks"]
            else:
                prompts_vicuna = prompts_4

            if "llama-2-7b-chat-hf.json" in files:
                prompts_llama = json.load(open(os.path.join(root, "llama-2-7b-chat-hf.json")))
                prompts_llama = prompts_llama["jailbreaks"]
            else:
                prompts_llama = prompts_4

            res_4 = [p["prompt"] for p in prompts_4]
            res_3_5 = [p["prompt"] for p in prompts_3_5]
            res_vicuna = [p["prompt"] for p in prompts_vicuna]
            res_llama = [p["prompt"] for p in prompts_llama]

            res = []

            for ids, (r1, r2, r3, r4) in enumerate(zip(res_4, res_3_5, res_vicuna, res_llama)):
                # print(r1, r2, r3)
                if r1:
                    res.append(r1)
                elif r2:
                    res.append(r2)
                elif r3:
                    res.append(r3)
                elif r4:
                    res.append(r4)
                else:
                    res.append(df.iloc[ids]["Goal"])

            attack_method = root.split('/')[-2]
            attack_method = "AIM" if "JBC" in attack_method else attack_method

            print(attack_method, [res[10]], '\n\n\n')

            df[attack_method] = res

    data = json.load(open(os.path.join('past_tense', 'tense_past.json')))['results']
    res = [''] * 100
    for d in data:
        p = d['results'][0]['request_reformulated']
        res[d['i_request']] = p

    df['past_tense'] = res

    data = json.load(open(os.path.join('past_tense', 'tense_future.json')))['results']
    res = [''] * 100
    for d in data:
        p = d['results'][0]['request_reformulated']
        res[d['i_request']] = p

    df['tense_future'] = res

    df.to_csv("jbb_expanded.csv", index=False)
