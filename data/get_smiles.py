import pubchempy as pcp
import dill

# 使用 PubChem 查询药物信息（可以使用药物名称或其他标识符）


nametosmiles = {}
# # 遍历查询结果，输出 SMILES 字符串
# for compound in compounds:
#     print(f"Compound ID: {compound.cid}, SMILES: {compound.isomeric_smiles}")


with open('drug_list.pkl', 'rb') as f:
    loaded_drug_list = dill.load(f)


# 检查文件是否存在且为空，如果是，则只在文件开头写入标题行

num = 0 
error_list =[]
error_num=0
for name in loaded_drug_list:
    num += 1
    print(f'{num}/{len(loaded_drug_list)}')
    compounds = pcp.get_compounds(name, 'name')
    if len(compounds)>0:  # 确保data_lines不为空
        nametosmiles[name.lower()] = compounds[0].isomeric_smiles
    else:
        error_num+=1
        error_list.append(name)
print(error_num)
dill.dump(nametosmiles, open('./NAMEtoSMILES_new.pkl', "wb"))
