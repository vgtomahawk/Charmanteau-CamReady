#from werkzeug.contrib.cache import SimpleCache
#cache=SimpleCache()
from flask import Flask
app = Flask(__name__)
from flask import request
from flask import render_template
from flask import _app_ctx_stack
from flask import g


@app.route("/test", methods=['GET', 'POST'] )
def test():
    print request.args
    inp = request.args.get('text')
    portmanteau_inputs = inp.split(',')
    #shakepeare_inputs = inp.strip()
    print portmanteau_inputs
    # TO DO: insert logic for portmanteau prediction / shakeeare style prediction
    return inp

@app.route("/portmanteau",methods=['GET'])
def hello():
    return render_template("hello.html")

@app.route("/portmanteau",methods=['POST'])
def hello_post():
    text=request.form['text']
    text2=request.form['text2']
    text=text.encode('utf8')
    text2=text2.encode('utf8')
    if len(text)==0:
        text="alpha"
    if len(text2)==0:
        text2="beta"
    answers=bed.query(text,text2,predictor)
    answers=[str(i)+". "+answer for i,answer in enumerate(answers)]
    headerString="<h1>The top 5 suggestions for a portmanteau are as below</h1>"
    answerString=headerString+"<br/>".join(answers)
    return answerString

if __name__=="__main__":
    from pre import *
    app.run(host="0.0.0.0",debug=False)
