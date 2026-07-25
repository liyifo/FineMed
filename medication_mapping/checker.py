import dill
import ast
import pandas as pd
import os
import openai

openai.api_base=""
openai.api_key=""





data = dill.load(open('./data_final.pkl', "rb"))

print(len(data))
data['checker'] = None
error = []
total_suc = 0
total_usage = 0



def evaluation(result, diag, proc):
    total = 0
    if len(result) != diag:
        print(f'diag num rerror! {len(result)}/{diag}')
        return False
    
    for i in result:
        total += len(i[1])
    if total != proc:
        print(f'proc num rerror! {total}/{proc}')
        return False
    

    return True


def generate_prompt(medication_list):
    template = f"""The following is a list of multiple triples, each in the form of "[diagnosis, procedure list, laboratory event list]", where diagnosis and procedure are represented by ICD-9 codes:
{medication_list}

You can only answer "Yes" or "No". You output "Yes" only if the following two conditions are met, otherwise output "No", please return it in plain text.

1. The triplets in the list conform to medical logic. In each triplet, procedures and laboratory events may be related to the corresponding diagnosis (procedures are used to treat diagnoses, and laboratory events reflect the diagnosed condition).
2. The list given to you conforms to the python list format.
"""
    return template


def requestGPT(index, sub_label):
    global total_usage, total_suc, error

    prompt = generate_prompt(sub_label)

    #return prompt
    print('begin query')
    completion = openai.ChatCompletion.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a skilled clinical coding professional."},
            {"role": "user", "content": prompt}
            ]
    )

    

    
    try:
        total_usage += completion['usage']['prompt_tokens'] + completion['usage']['completion_tokens'] + completion['usage']['total_tokens']
        answer = completion.choices[0].message.content
        # 尝试将字符串转换为列表
        result = ast.literal_eval(answer)
        
        # 检查转换的结果是否为列表
        if isinstance(result, list) and evaluation(result, len(diag), len(proc)):
            total_suc += 1  # 成功转换为列表
            print('success!')
            return completion.choices[0].message.content
        else:
            error.append(index)  # 转换成功，但不是列表
            return None
    except Exception as e:
        # 转换失败
        print(e)
        error.append(index)
        return None

#print(requestGPT(0, data.at[0, 'ICD9_CODE'], data.at[0, 'PRO_CODE'], data.at[0, 'TEXT']))

for index, row in data.iterrows():
    if data.at[index, 'checker']==None:
        print(f'{index}/{len(data)}    suc:{total_suc}')
        ans = requestGPT(index, row['sub_label'])
        if ans == "No":
            data.at[index, 'checker'] = 1
            data.at[index, 'sub_label'] = None
        else:
            data.at[index, 'checker'] = 2

        dill.dump(obj=data, file=open('data_final.pkl', 'wb'))


print(f'total usage token:{total_usage}')
print(f'error:{error}')
#flag = len(data[0][0])


# total = 0
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         if  len(data[i][j])==5:
#             try:
#             ans = ''
#             data[i][j].append(ans)
#             total+=1

dill.dump(obj=data, file=open('data_final.pkl', 'wb'))



#print(total)

