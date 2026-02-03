import dill
import ast
import pandas as pd
import os
import openai

openai.api_base=""
openai.api_key=""





data = dill.load(open('./data_final.pkl', "rb"))

print(len(data))
data['sub_label'] = None
error = []
total_suc = 0



def evaluation(result, diag, proc, med):
    if len(result) != med:
        print(f'med num rerror! {len(result)}/{med}')
        return False
    
    # for i in result:
    #     if len(i[1]) == 0 and len(i[2]) == 0:
    #         print(f'Empty med: {i[0]}')
    #         return False
    

    return True


def generate_prompt(diag, proc, med, hosp_course):
    template = f"""As a skilled clinical doctor, it is your responsibility to determine which diagnoses and procedures are being treated with each medication in a given prescription. Here are the things you need to pay attention to:

1. The diagnosis and treatment procedures are provided to you in the form of an ICD9 coding list.
2. The drugs in the prescription must be prescribed for at least one disease or diagnosis, therefore, you must treat at least one diagnosis or procedure for each drug.
3. The Brief Hospital Course chapter in the clinical notes may contain some facts about targeted medication, which you can refer to.

Below I will provide you with a list of diagnoses, procedures, and medications, as well as the Brief Hospital Course section of the clinical notes for this prescription:
{len(diag)} Diagnosis list: {diag}
{len(proc)} Procedure list: {proc}
{len(med)} Medication list: {med}
Brief Hospital Course: {hosp_course}

The result you give should be a list. Where the number of sublists is equal to the number of drugs, and each sublist is a correspondence between a drug and a diagnosis or procedure. You should not output any other text unrelated to the list. Please return the list without code blocks in plain text format. Return result example : [['drug1', ['diagnosis1'], ['procedure1']], ['drug2', ['diagnosis1', 'diagnosis2'], []]]
"""
    return template


def requestGPT(index, diag, proc, med, sections):
    global total_suc, error


    prompt = generate_prompt(diag, proc, med, sections['brief hospital course'])

    #return prompt
    print('begin query')
    completion = openai.ChatCompletion.create(
        model="gpt-4o-2024-05-13",
        messages=[
            {"role": "system", "content": "You are a skilled clinician."},
            {"role": "user", "content": prompt}
            ]
    )

    

    
    try:
        answer = completion.choices[0].message.content
        # 尝试将字符串转换为列表
        print(answer)
        result = ast.literal_eval(answer)
        
        
        # 检查转换的结果是否为列表
        if isinstance(result, list) and evaluation(result, len(diag), len(proc), len(med)):
            total_suc += 1  # 成功转换为列表
            print('success!')
            return result
        else:
            error.append(index)  # 转换成功，但不是列表
            #print(result)
            return None
    except Exception as e:
        # 转换失败
        print(e)
        #print(result)
        error.append(index)
        return None

#print(requestGPT(0, data.at[0, 'ICD9_CODE'], data.at[0, 'PRO_CODE'], data.at[0, 'TEXT']))

for index, row in data.iterrows():
    if data.at[index, 'sub_label']==0 or data.at[index, 'sub_label']==None:
        print(f'{index}/{len(data)}    suc:{total_suc}')
        ans = requestGPT(index, row['ICD9_CODE'], row['PRO_CODE'], row['DRUG'], row['sections'])
        data.at[index, 'sub_label'] = ans
        if ans != None:
            dill.dump(obj=data, file=open('data_final.pkl', 'wb'))


print(f'len:{len(error)} error:{error}')
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

