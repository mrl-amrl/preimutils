import json


class LabelHandler:
    def __init__(self,json_path):
        self.json_path = json_path
        
    def json_id_label_dic(self):
        with open(self.json_path) as f:
            data = json.load(f)

        # print(data)
        data = dict(data)
        keys = data.keys()
        categori_id_name = {}
        for key in keys:
            key = int(key)
            categori_id_name[key] = data['{}'.format(key)]
        
        return categori_id_name

    def json_label_id_dic(self):
        with open(self.json_path) as f:
            data = json.load(f)
        data = dict(data)
        keys =[int(key) for key in data.keys()]
        values = data.values()
        return (dict(zip(values,keys)))
    
    def json_label_array(self):
        with open(self.json_path) as f:
            data = json.load(f)
        data = dict(data)
        values = data.values() 
        return list(values)

