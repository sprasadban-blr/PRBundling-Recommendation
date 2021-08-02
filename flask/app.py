from flask import Flask, request, jsonify, send_file,render_template
from prbundle import getPRs,generateMLInterpretability
from flask_cors import CORS
import pandas as pd
import json
import os
import pickle
# import imgkit


app = Flask(__name__)
CORS(app)

originaldata=[]

@app.route('/', methods=['GET'])
def date_input():
    startdate = request.args.get('startdate')
    enddate = request.args.get('enddate')
    print(startdate,enddate)
    if startdate != 0 and enddate != 0 and startdate < enddate:
        global originaldata
        result,originaldata = getPRs(startdate, enddate)
        print(type(result),result)
        data = []
        for key, value in result.items():
            obj = {}
            obj["cluster"]= key
            obj["total"]= len(value)
            data.append(obj)
        # print(result)
        print(data)
        return jsonify(data)

def convertToJSON(keys,l):
    result=[]
    for items in l:
        obj={}
        for i in range(len(keys)):
            obj[keys[i]] = items[i]
        result.append(obj)
    return result

@app.route('/clusterdata',methods=['GET'])
def getClusterData():
    cluster = int(request.args.get('cluster'))
    print(type(cluster))
    originaldata = pickle.load(open('original_data.sav', 'rb'))
    originaldata['cluster'] = originaldata['cluster'].astype(int)
    print(originaldata.info())
    print(originaldata.loc[originaldata['cluster']==cluster])
    result = {}
    result['chart1'] = pd.DataFrame(originaldata['EKORG'].value_counts().rename_axis('unique_values').reset_index(name='counts'))
    result['chart2'] =  pd.DataFrame(originaldata['cluster'].value_counts().rename_axis('unique_values').reset_index(name='counts'))
    result['table'] =  pd.DataFrame((originaldata.loc[originaldata['cluster']==cluster]))
    result['chart1'] = convertToJSON(result['chart1'].columns,result['chart1'].values.tolist())
    result['chart2'] = convertToJSON(result['chart2'].columns, result['chart2'].values.tolist())
    result['table'] = convertToJSON(result['table'].columns,result['table'].values.tolist())
    print(result)
	
    return result


@app.route('/interpretResult',methods=['POST'])
def getInterpretation():
    # jsonData = {"MANDT": "100", "BANFN": "0522981722", "BNFPO": "00060", "FLIEF": "0001789582", "EKORG": "JP01", "cluster": 0}
    
    # return send_file('download.png', mimetype='image/png')
    jsonInput = json.dumps(request.json, indent = 4)
    print(jsonInput)
    
    generateMLInterpretability(jsonInput)

    result={"result":True}
    
    return result

@app.route('/loadInterpretaibility/<banfn>', methods=['GET'])
def getLimeResult(banfn):
    cwd = os.getcwd()
    # imgkit.from_file(cwd+'\loaded_lime_instance.html', 'out.jpg')
    return send_file(cwd+'\loaded_lime_instance.html')
    # return send_file(cwd+'\loaded_lime_instance.jpg', mimetype='image/jpg')

if __name__ == '__main__':
    app.run(debug=True)