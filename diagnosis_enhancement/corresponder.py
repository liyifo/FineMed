import dill
import ast
import pandas as pd
import os
import openai

openai.api_base=""
openai.api_key=""





data = dill.load(open('./data_final.pkl', "rb"))

print(len(data))
data['cluster'] = None
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


def generate_prompt(diag, proc, test, chief, history, exam):
    template = f"""As a skilled clinical coding professional, it is your responsibility to cluster procedure ICD-9 codes and laboratory test results to the appropriate diagnosis ICD-9 codes. Here's what you need to be aware of:

1. Each diagnosis code corresponds to a category.
2. The principle of clustering is “is it associated with the same disease” to ensure that the procedure is used to treat the diagnosed disease and that the laboratory test result is related to the diagnosed disease.
3. Each procedure code must and can only correspond to one diagnosis code.
4. Each diagnosis code must appear in the results.
5. The same laboratory test result may be categorized to multiple different diagnosis codes. 
6. The number of diagnosis codes and procedure codes output must be equal to the number of codes I gave you.
7. There may be some evidence in some chapters of the clinical record for your reference.

Below, I'll provide you with the code to cluster and some sections from the clinical note:
{len(diag)} diagnosis codes corresponding to the number of categories in the results: {diag}
{len(proc)} procedure codes that must be classified only once: {proc}
{len(test)} laboratory test results that can be categorized into multiple categories: {test}
Chief Complaint: {chief}
History of Present Illness: {history}
Physical Exam: {exam}

The format of the answer you give should be a list with the first level representing the clustered results for the different diagnoses, the first item on the second level being the diagnosis code for that cluster, the second item on the second level being the list of procedure codes for that cluster, and the third item on the second level being the list of laboratory test results. Do not output anything else. Please return the list without code blocks in plain text format. Example:
'''
[["E915", ["45.13"], []],
["038.8", ["00.14"], ["White Blood Cells: 30.5 K/uL"]],
["250.00", [], ["Glucose: 200mg/dL"]]]
'''
"""
    return template


def requestGPT(index, diag, proc, test, sections):
    global total_usage, total_suc, error

    prompt = generate_prompt(diag, proc, test, sections['chief complaint'], sections['history of present illness'], sections['physical exam'])

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
    if data.at[index, 'cluster']==0 or data.at[index, 'cluster']==None:
        print(f'{index}/{len(data)}    suc:{total_suc}')
        ans = requestGPT(index, row['ICD9_CODE'], row['PRO_CODE'], row['test_text'], row['sections'])
        data.at[index, 'cluster'] = ans
        if ans != None:
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

