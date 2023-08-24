import json
def Create_json(test_predict,file_name):
    data_list = test_predict.tolist()
    data = {}
    for i, value in enumerate(data_list):
        key = str(i)
        value = data_list[i][0]
        data[key] = value
# 将JSON字符串写入文件
    with open("file_name", "w") as file:
        json.dump(data, file)
