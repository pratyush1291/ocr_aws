import os
import pandas as pd
total_lastnames=[]
folder_path=r'D:\elvote\Maharashtra\maharashtra mob no'
filenames=os.listdir(folder_path)


for file in filenames:
    folder=os.path.join(folder_path,file)
    print(folder)

    data=pd.read_excel(folder)
    data['LASTNAME_EN'] = data['LASTNAME_EN'].str.upper()
    unique_n = data['LASTNAME_EN'].unique()
    unique_names = pd.DataFrame(unique_n)
    total_lastnames.append(unique_names)




final_df = pd.concat(total_lastnames, ignore_index=True)
final_df.to_excel('total_unique_surnames.xlsx', index=False)
